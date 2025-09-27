import torch
from torch import nn
from util.misc import nested_tensor_from_videos_list, NestedTensor
from models.CMT_adapter import CMT_adapter
from hydra import compose, initialize
import spacy
from models.sam2.modeling.sam2_utils import preprocess
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os
import py3_wget
from models.conditional_memory_encoder import ConditionalMemoryEncoder
from fairseq.models.roberta import RobertaModel
from models.model_utils import BackboneOutput, DecoderOutput, get_same_object_labels
from transformers import RobertaTokenizerFast


class SAMWISE(nn.Module):
    def __init__(self,
                 image_encoder_embed_dim,
                 text_encoder,
                 text_encoder_embed_dim,
                 fusion_stages_txt,
                 fusion_stages,
                 image_size,
                 sam,
                 conditional_memory_encoder,
                 adapter_dim,
                 args):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.sam = sam
        self.conditional_memory_encoder = conditional_memory_encoder
        if args.motion_prompt:
            # load nlp dict to identify verbs
            self.nlp_dict = spacy.load('en_core_web_sm')
        self.motion_prompt = args.motion_prompt

        # build Cross Modal Temporal adapter
        self.cmt_adapters = nn.ModuleList()
        for i in range(len(fusion_stages)):
            self.cmt_adapters.append(CMT_adapter(
                                        in_channels_vis=image_encoder_embed_dim[fusion_stages[i]-1],
                                        in_channels_txt=text_encoder_embed_dim,
                                        adapter_channels=adapter_dim,
                                        HSA_patch_size=args.HSA_patch_size[i] if len(args.HSA_patch_size)>1 else args.HSA_patch_size[0],
                                        args=args))

        self.memory_bank = {} # to store all frames memory

        self.fusion_stages_txt = fusion_stages_txt
        self.fusion_stages_vis = sam.image_encoder.trunk.stage_ends
        self.fusion_stages = fusion_stages
        self.image_size = image_size

        self.use_cme_head = args.use_cme_head
        self.cme_decision_window = args.cme_decision_window # minimum number of frames between each CME application
        self.switch_mem = args.switch_mem

        self.device = args.device


    def forward(self, samples, captions, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
               - captions: list[str]
               - targets:  list[dict]; during training contains masks, during inference frame Id info
            It returns a dict with the following elements:
               - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]
        """

        # samples: tensor B*T, C, H, W
        backbone_output: BackboneOutput = self.compute_backbone_output(samples, captions)
        B, T = backbone_output.B, backbone_output.T
        outputs = {"masks": []}

        for video_record in range(B):
            if self.training or T==1: # T == 1 for pre-training, no propagation from memory bank
                self.memory_bank, self.last_frame_cme_applied = {}, 0
            elif targets[0]['frame_ids'][0] == 0:  # it's the first frame of a new video
                self.memory_bank, self.last_frame_cme_applied = {}, 0

            for frame_idx in range(T):
                idx = video_record * T + frame_idx
                # use relative IDX in the clip
                if self.training or T==1:  # T == 1 for pre-training, no propagation from memory bank
                    memory_idx = frame_idx
                # use absolute IDX in the video
                else:
                    memory_idx = targets[0]['frame_ids'][frame_idx]

                current_vision_feats = backbone_output.get_current_feats(idx)
                decoder_out_w_mem: DecoderOutput = self.compute_decoder_out_w_mem(backbone_output, idx, memory_idx,
                                                                                  self.memory_bank)

                if self.use_cme_head:
                    # wait at least cme_decision_window frames between 2 CME applications
                    if memory_idx - self.last_frame_cme_applied >= self.cme_decision_window-1 and memory_idx>self.cme_decision_window:
                        # memory-less prediction
                        decoder_out_no_mem_cme: DecoderOutput = self.compute_decoder_out_no_mem(backbone_output, idx)
                        pred_cme_logits = self.conditional_memory_encoder(decoder_out_w_mem.obj_ptr.detach(),
                                                                          decoder_out_no_mem_cme.early_obj_ptr.detach())

                        if pred_cme_logits.argmax().item() == 1 and not self.training:  # not training and switch
                            decoder_out_w_mem = self.apply_decision(decoder_out_w_mem, decoder_out_no_mem_cme)
                            self.last_frame_cme_applied = memory_idx

                        if self.training:
                            # cme_label indicates whether memory features and memory-less features point to same object
                            cme_label = get_same_object_labels(decoder_out_w_mem.masks.detach().cpu(),
                                                               decoder_out_no_mem_cme.masks.detach().cpu(),
                                                               decoder_out_no_mem_cme.object_score_logits.detach()).item()

                            if 'pred_cme_logits' not in outputs:
                                outputs['pred_cme_logits'] = []
                                outputs["cme_label"] = []
                            outputs["pred_cme_logits"].append(pred_cme_logits)
                            outputs["cme_label"].append(cme_label)

                mem_dict_w_mem = self.compute_memory_bank_dict(decoder_out_w_mem, current_vision_feats, backbone_output.feat_sizes)
                self.memory_bank[memory_idx] = mem_dict_w_mem
                outputs["masks"].append(decoder_out_w_mem.masks)

        masks = torch.cat(outputs["masks"])
        if self.training:
            return outputs
        else:
            return {"pred_masks": masks.squeeze(1)}


    @staticmethod
    def preprocess_visual_features(samples, image_size):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)
        samples, masks = samples.decompose()
        B, T, C, H, W = samples.shape
        samples = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([preprocess(x, image_size) for x in samples], dim=0)
        BT = (B, T)
        return samples, BT, orig_size

    def preprocess_text_features(self, captions):
        batch_encoding_text = self.tokenizer(captions, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(batch_encoding_text['input_ids']).to(self.device)
        attention_mask = torch.tensor(batch_encoding_text['attention_mask']).eq(0).to(self.device)
        text_encoder = self.text_encoder.model.encoder.sentence_encoder
        has_pads = (torch.tensor(input_ids.device.type == "xla") or attention_mask.any())
        x, encoder_embedding = text_encoder.forward_embedding(input_ids, None)
        x = x * (1 - attention_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x))
        txt = x.transpose(0, 1)  # B x T x C -> T x B x C
        return txt, attention_mask, input_ids
    
    def compute_backbone_output(self, samples, captions):
        samples, BT, orig_size = self.preprocess_visual_features(samples, self.image_size)
        txt, attention_mask, input_ids = self.preprocess_text_features(captions)

        B, T = BT

        if self.motion_prompt:
            vis_outs, state, txt = self._early_fusion_stage(T, samples, txt, attention_mask)
            motion_prompts = self.extract_motion_prompts(captions, input_ids)
            motion_state = [txt_i[motion_prompts[i].bool()] for i, txt_i in enumerate(txt)]
            motion_state = torch.cat(motion_state).repeat_interleave(T, 0)
        else:
            vis_outs, state = self._early_fusion_stage(T, samples, txt, attention_mask)
            motion_state = torch.empty(1)

        # forward FPN
        backbone_out = self._forward_fpn(vis_outs)
        _, vision_feats, vision_pos_embeds, feat_sizes = self.sam._prepare_backbone_features(backbone_out)
        out = BackboneOutput(B, T, orig_size, vision_feats, vision_pos_embeds, feat_sizes, state, motion_state)
        return out

    def compute_decoder_out_w_mem(self, backbone_out: BackboneOutput, idx: int, memory_idx: int, memory_bank: dict):
        current_vision_feats = backbone_out.get_current_feats(idx)
        current_vision_pos_embeds = backbone_out.get_current_pos_embeds(idx)
        # take only the highest res feature map
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)
        
        pix_feat_with_mem = self._prepare_memory_conditioned_features(
            # it's absolute frame ID in eval, relative in the clip during train
            frame_idx=memory_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:],
            num_frames=memory_idx+1, # how many obj_ptr to take from mem
            memory_bank=memory_bank
        )
        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            text_inputs=backbone_out.state[idx:idx+1],
            motion_inputs=backbone_out.motion_state[idx:idx+1] if self.motion_prompt else None,
            high_res_features=high_res_features,
        )
        decoder_out.compute_mask(self.image_size, backbone_out.orig_size[idx])
        return decoder_out

    def compute_decoder_out_no_mem(self, backbone_out: BackboneOutput, idx: int):
        current_vision_feats = backbone_out.get_current_feats(idx)
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)

        pix_feat_no_mem = current_vision_feats[-1:][-1] + self.sam.no_mem_embed
        pix_feat_no_mem = pix_feat_no_mem.permute(1, 2, 0).view(1, 256, 64, 64)
        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_no_mem,
            text_inputs=backbone_out.state[idx:idx+1],
            high_res_features=high_res_features,
        )
        decoder_out.compute_mask(self.image_size, backbone_out.orig_size[idx])
        return decoder_out

    def compute_memory_bank_dict(self, decoder_out: DecoderOutput, current_vision_feats, feat_sizes):
        maskmem_features, maskmem_pos_enc = self.sam._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=decoder_out.high_res_masks,
            is_mask_from_pts=False,
        )

        memory_dict = {"maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": decoder_out.low_res_masks,
            "obj_ptr": decoder_out.obj_ptr,
        }
        return memory_dict

    def extract_motion_prompts(self, captions, input_ids):
        docs = [self.nlp_dict(x) for x in captions]
        motion_map = torch.zeros(size=input_ids.shape).to(input_ids.device)
        encoded_input = self.tokenizer(captions, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True, padding=True)
        for caption_index, doc in enumerate(docs):
            roberta_tokens = self.tokenizer.convert_ids_to_tokens(encoded_input.input_ids[caption_index])
            roberta_offsets = encoded_input['offset_mapping'][caption_index]
            for index, (rt, offset) in enumerate(zip(roberta_tokens, roberta_offsets)):
                start, end = offset
                if rt != '<s>' and rt != '</s>':
                    for token in doc:
                        if token.idx <= start and (token.idx + len(token.text)) >= end:
                            if token.pos_ == 'VERB':
                                motion_map[caption_index][index] = 1
        return motion_map
    
    def apply_decision(self, decoder_out_w_mem: DecoderOutput, decoder_out_no_mem: DecoderOutput):
        high_res_masks = decoder_out_w_mem.high_res_masks
        hm = decoder_out_no_mem.high_res_masks
        if self.switch_mem == 'all_mask':
            high_res_masks = hm
        elif self.switch_mem == 'reweight':
            high_res_masks[hm > 0] = hm[hm > 0]*10
        elif self.switch_mem == 'avg':
            high_res_masks = (high_res_masks + hm) / 2
        decoder_out_w_mem.high_res_masks = high_res_masks
        return decoder_out_w_mem

    @staticmethod
    def forw_layer_list(start, end, layers, x, attention_mask=None):
        for idx in range(start, end):
            if attention_mask is not None:
                x = layers[idx](x, encoder_padding_mask=attention_mask)
            else:
                x = layers[idx](x)
        return x

    def _early_fusion_stage(self, T, samples, txt, attention_mask):
        vis = self.sam.image_encoder.trunk.patch_embed(samples)
        vis = vis + self.sam.image_encoder.trunk._get_pos_embed(vis.shape[1:3])
        vis_outs = []
        fusion_stages_vis = [x+1 for x in self.fusion_stages_vis]

        fusion_vis = fusion_stages_vis.copy()
        fusion_vis.insert(0, 0)
        fusion_txt = self.fusion_stages_txt.copy()
        fusion_txt.insert(0, 0)
        fusion_txt.insert(1,1)
        for i, (i_v, i_t) in enumerate(zip(fusion_vis[:-1], fusion_txt[:-1])):
            vis = self.forw_layer_list(i_v, fusion_vis[i+1], self.sam.image_encoder.trunk.blocks, vis)
            txt = self.forw_layer_list(i_t, fusion_txt[i+1], self.text_encoder.model.encoder.sentence_encoder.layers, txt, attention_mask)
            if i in self.fusion_stages:
                v = vis.clone()
                t = txt.clone()
                v, t = self.cmt_adapters[self.fusion_stages.index(i)](v.permute(0, 3, 1, 2), T, t)
                vis = vis + v.permute(0, 2, 3, 1)
                txt = txt + t

            vis_outs.append(vis.permute(0, 3, 1, 2))

        txt = txt.permute(1, 0, 2)  # LND -> NLD
        state = txt[:,0]
        if T > 1:
            state = state.repeat_interleave(T, 0)

        if self.motion_prompt:
            return vis_outs, state, txt
        return vis_outs, state

    def _forward_fpn(self, vis_outs):
        features, pos = self.sam.image_encoder.neck(vis_outs)

        # Discard the lowest resolution features
        features, pos = features[: -1], pos[: -1]
        image_embedding = features[-1]

        backbone_out = {
            "vision_features": image_embedding,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        backbone_out["backbone_fpn"][0] = self.sam.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        return backbone_out

    # simplify SAM2 _prepare_memory_conditioned_features method
    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        num_frames,
        memory_bank
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)   # batch size on this frame: always 1
        C = self.sam.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.sam.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        # Step 1: condition the visual features of the current frame on previous memories
        if frame_idx != 0:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            t_pos_and_prevs = []
            chosen_frames = [] # just for debug
            for t_pos in range(1, self.sam.num_maskmem):
                t_rel = self.sam.num_maskmem - t_pos  # how many frames before current frame
                prev_frame_idx = frame_idx - t_rel
                chosen_frames.append(prev_frame_idx) # just for the debug print below
                t_pos_and_prevs.append((t_pos, memory_bank.get(prev_frame_idx, None)))
            # print([tpp[0] for idd, tpp in enumerate(t_pos_and_prevs) if tpp[1] is not None])

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                feats = prev["maskmem_features"]
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1]
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                tpos_enc_id = self.sam.num_maskmem - t_pos - 1
                # TODO: this is a hack to use more frames than the pretrained maskmem_tpos_enc
                # allows for. It uses the same encoding for all older frames; decide what to do
                # with this
                tpos_enc_id = min(tpos_enc_id, self.sam.maskmem_tpos_enc.shape[0] - 1)
                maskmem_enc = (
                    maskmem_enc + self.sam.maskmem_tpos_enc[tpos_enc_id]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.sam.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.sam.max_obj_ptrs_in_encoder)
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                pos_and_ptrs= []
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx - t_diff
                    if t < 0 or t >= num_frames:
                        break
                    out = memory_bank.get(t, None)
                    if out is not None:
                        pos_and_ptrs.append((t_diff, memory_bank[t]["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.sam.mem_dim)
                    if self.sam.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.sam.mem_dim, self.sam.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.sam.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat_with_mem = current_vision_feats[-1] + self.sam.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem


        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.sam.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem


from models.path_utils import ROBERTA_WEIGHTS_PATH, SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from models.path_utils import get_roberta_weights

def build_samwise(args):
    if not os.path.isdir(ROBERTA_WEIGHTS_PATH):
        get_roberta_weights()
    # build text encoder
    roberta = RobertaModel.from_pretrained(ROBERTA_WEIGHTS_PATH, checkpoint_file='model.pt')
    text_encoder_embed_dim = roberta.model.encoder.lm_head.dense.out_features

    sam2_weights, sam2_config = SAM2_PATHS_CONFIG[args.sam2_version]
    if not os.path.isfile(sam2_weights):
        print(f"Downloading SAM2-{args.sam2_version}")
        py3_wget.download_file(SAM2_WEIGHTS_URL[args.sam2_version], sam2_weights)
    
    # build sam2 image encoder and decoder
    with initialize(version_base=None, config_path="sam2", job_name="test_app"):
        cfg = compose(config_name=sam2_config, overrides=[f"++model.motion_prompt={args.motion_prompt}",
                                                               f"++model.text_encoder_embed_dim={text_encoder_embed_dim}"])
        OmegaConf.resolve(cfg)
        cfg.model.pred_obj_scores = not args.disable_pred_obj_score
        cfg.model.pred_obj_scores_mlp = not args.disable_pred_obj_score
        cfg.model.fixed_no_obj_ptr = not args.disable_pred_obj_score
        sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(sam2_weights, map_location="cpu")["model"]
    sam.load_state_dict(state_dict, strict=False)
    sam_embed_dim = cfg.model.image_encoder.neck.backbone_channel_list[::-1][1:]

    # build Conditional Memory Encoder
    conditional_memory_encoder = ConditionalMemoryEncoder(sam.hidden_dim)

    ## Samwise
    model = SAMWISE(
        image_encoder_embed_dim=sam_embed_dim,
        text_encoder=roberta,
        text_encoder_embed_dim=text_encoder_embed_dim,
        fusion_stages_txt=args.fusion_stages_txt,
        fusion_stages=args.fusion_stages,
        image_size=sam.image_size,
        sam=sam,
        conditional_memory_encoder=conditional_memory_encoder,
        adapter_dim= args.adapter_dim,
        args=args
    )


    # freeze all the weights except CMT adapter and Conditional Memory Encoder
    for param_name, param in model.named_parameters():
        if 'adapter' not in param_name and 'conditional_memory_encoder' not in param_name and 'project_text' not in param_name:
            param.requires_grad = False

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return model
