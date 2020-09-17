# from imageio import mimsave, imread
from glob import glob
from natsort import natsorted
import os
from PIL import Image, ImageDraw  # , ImageFont
import gif


def make_gif_from_dir(gif_fpath, img_dir, frame_length=0.5, font_size=20,
                      delete_images=False):
    """
    Makes a gif from images stored in a directory
    """
    img_names = natsorted([os.path.basename(f)
                           for f in glob('{}/*.png'.format(img_dir))])

    images = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)  # load image

        if font_size is not None:
            # fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', font_size)
            txt = img_name.split('.')[0].replace('_', ' ')
            d = ImageDraw.Draw(img)
            # d.text((10, 10), txt, font=fnt, fill=(255, 0, 0))
            d.text((10, 10), txt, fill=(255, 0, 0))
        images.append(img)

        if delete_images:
            os.remove(img_path)

    images[0].save(gif_fpath,
                   save_all=True,
                   append_images=images[1:],
                   duration=frame_length,
                   loop=1)


def make_gif(frame_iter, fpath, duration=100):
    """

    Parameters
    ----------
    frame_iter: iterable

    fpath: str

    duration: int

    Example
    -------
    def plot_func(power):

      plt.figure(figsize=(5, 5))
      xvals = np.arange(5)
      yvals = xvals ** power

      plt.plot(xvals, yvals, marker='.')


    def arg_iter():
        for p in range(5):
            yield {'power': p}
    """
    frames = [frame for frame in frame_iter]
    gif.save(frames, fpath, duration=duration)


def get_frame_iter(plot_func, kwarg_iter):

    @gif.frame
    def _plot_func(kwargs):
        plot_func(**kwargs)

    for kws in kwarg_iter:
        yield _plot_func(kws)
