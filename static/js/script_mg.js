function select_one_scene(scene_id) {
  var scenes = document.getElementsByClassName('scene');

  for (var i = 0; i < scenes.length; i++) {
    scenes[i].style.display = 'none';
  }

  document.getElementById(scene_id).style.display = 'block';

  const raw_video = document.querySelector(`#${scene_id} .raw-video`);
  raw_video.style.display = 'block';
  disappear_annotated_video();
  disable_all_buttons()


  const buttons = document.querySelectorAll(`#${scene_id} .caption-button`);
  for (var i = 0; i < buttons.length; i++) {
        buttons[i].removeAttribute('disabled');
    }

    var thumbnails = document.getElementsByClassName('thumbnail-btn');
    for (var i = 0; i < thumbnails.length; i++) {
        thumbnails[i].classList.remove('selected');
    }
  document.getElementById('thumb-' + scene_id).classList.add('selected');

}

function slide_left() {
  slider_window = document.getElementById('thumbnails-scroll');
  slider_window.scrollLeft = 0;
}

function slide_right() {
  slider_window = document.getElementById('thumbnails-scroll');
  slider_window.scrollLeft += 1000;
}

function disable_all_buttons()
    {
        var caption_buttons = document.getElementsByClassName('caption-button');
        for (var i = 0; i < caption_buttons.length; i++) {
            caption_buttons[i].style.opacity = 0.6;
            caption_buttons[i].style.border = '2px solid transparent'

        }
    }

function disappear_annotated_video()
    {
        var annotated_scenes = document.getElementsByClassName('annotated-scene');
        for (var i = 0; i < annotated_scenes.length; i++) {
            annotated_scenes[i].style.display = 'none';
        }
    }


function handleCaptionClick(path_raw_video, path_annotated_video) {
    disappear_annotated_video();
    disable_all_buttons();
    const raw_video = document.querySelector(`#${path_raw_video} .raw-video`);
    raw_video.style.display = 'none'
    const annotated_video = document.querySelector(`#${path_annotated_video}`);

    annotated_video.style.display = 'block'

    const clicked_caption = event.target;
    clicked_caption.style.opacity = 1;
    clicked_caption.style.border = '4px solid #ffcc00'

}


document.addEventListener('DOMContentLoaded', function() {
    disable_all_buttons();
  select_one_scene("path5");
  disappear_annotated_video();
});

