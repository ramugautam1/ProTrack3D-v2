# ProTrack3D

ProTrack3D finds protein clusters in 3D time-lapse images, follows each one through time, and tells you when they split, merge, show up and disappear. Then it makes the plots.

You give it a movie. It gives you masks, tracks, family trees and tables.

It runs on Windows, Mac and Linux. It uses your GPU if you have one (NVIDIA, or an Apple Silicon Mac). Without a GPU it still works, just slower.

## What you need

- A computer with 8 GB of RAM at the very least. 16 GB is better for long movies.
- Free disk space. A lot. See "Disk space" below before you start a long movie.
- Miniforge. It is free. Get it here: https://github.com/conda-forge/miniforge
- An NVIDIA card if you want to train your own model. You do not need one to use the models we provide.

## Install

1. Install Miniforge. Click through the installer, keep the defaults.
2. Get ProTrack3D. On this page click **Code**, then **Download ZIP**, and unzip it somewhere you will find it again. If you use git:

```bash
git clone https://github.com/ramugautam1/ProTrack3D-v2.git
```

3. Start it.
   - **Windows:** double-click `ProTrack3D_app.bat`
   - **Mac:** double-click `ProTrack3D_app.command`
   - **Linux:** run `./ProTrack3D_app.sh` in a terminal

The first start takes a few minutes. It is downloading and setting things up. It only does this once. After that it opens in a few seconds.

On a Mac, the first time, macOS may refuse to open the file because it came from the internet. If it does, open System Settings > Privacy & Security, scroll down, and press **Open Anyway**. Or open Terminal, drag the file into it, and press Return.

4. Get the models. In the app go to **Models** and press **Download published models**. That is about 500 MB.

That is it. You are ready.

## Updating

You do not need the terminal for this.

1. Go to **Home**. At the bottom there is a card called **Updates**. The app looks for a newer version by itself each time it starts; press **Check for updates** to look again.
2. If there is one, press **Update now**. It takes a few seconds.
3. Press **Restart ProTrack3D**.

Your models, images and results are not touched. Only the program files are replaced. Wait for running jobs to finish first; the app will tell you if you forget.

If you changed program files yourself, the update may refuse (a git copy) or replace your changes (a ZIP copy). Keep your own edits somewhere else.

## Your images

- One image per file. `.tif` works best. `.nii` also works.
- A movie is `TZYX`: time, then z, then y, then x. A single timepoint is `ZYX`.
- One channel only. If your file has two channels, split it first (Fiji: Image > Color > Split Channels).
- Any width and height is fine. You do not need multiples of 32 any more.

Pick the model that matches your protein. A Bazooka model on Bazooka images. If nothing matches, try the closest one and look at the result. If it is bad, train your own (see below).

## Run everything at once

This is what most people want.

1. Go to **Pipeline**.
2. Drop your image on the page. Or a whole folder of images.
3. Pick the model.
4. Set the first and last timepoint for each image if you do not want the whole movie.
5. Press **Run pipeline**.

It does segmentation, then tracking, then analysis, one image after another. Go to **Jobs** to watch. You can cancel any time. Cancel really stops it.

If you want to see exactly what it is doing, open the console (the small terminal icon, bottom left).

## Or one step at a time

- **Segmentation** finds the objects in each timepoint.
- **Tracking** follows them through time. It needs the segmentation first.
- **Analysis** makes the plots and tables. It needs the tracking first.
- **Family tree** draws the history of the objects you ask for.

Each page fills in what it can. Drop the image and it finds the earlier results by itself if they are next to the image.

## Where the results go

Next to your image, unless you choose another folder. For an image called `movie.tif`:

```
movie.tif
movie_SEG.tif                         the mask, same shape as your image
movie_SegmentationOutput/FC-DenseNet/
    CombinedSO/CombinedSO.nii         the mask again
    CombinedSO/CombinedSOLabeled.nii  every object numbered, per timepoint
    Features/                         what tracking uses. Big.
movie_TrackingOutput/
    TrackedCombined.nii               every object with its ID through time
    split_list.csv, merge_list.csv    who split from whom, who merged into whom
    Sheet3.csv                        every object: first and last timepoint, parent
    eventObjectCount.csv              counts per timepoint
    Analysis1-50/                     plots and tables
    FamilyTrees/                      family trees, if you drew any
```

Your original image is never changed.

## Looking at results

**Viewer.** Drop the image (`.tif`, `.nii` or `.czi`). If the file has several channels, pick the channel at the bottom. Scroll the mouse wheel to move through z. Drag the time slider or press play. Choose "Tracked objects" to see every object in its own colour. Click an object and its whole family lights up.

The panel on the right is brightness and contrast, the way ImageJ does it. The picture on top is the histogram of the slice you are looking at; the line over it shows what is drawn black and what is drawn white. Move **Minimum** and **Maximum**, or **Brightness** and **Contrast**, or type the two numbers. **Auto** fits the range to the slice; press it again to cut deeper. **Reset** shows the whole range. Each channel keeps its own setting. This only changes what you see. Your data and the measurements are not changed.

**Save image…** writes the image with that brightness and contrast put into the pixel values (what ImageJ calls Apply). It asks: a copy, or replace the original. A copy gets the name of the image with `_BC` added; change it if you like. All timepoints and slices are saved. A `.czi` is saved as a `.tif` copy. Careful with Replace: the measured intensities in that file are gone for good, so do not replace an image you still want to measure.

**Lineage.** Drop the `_TrackingOutput` folder. Every object is a row. Dots are timepoints. Blue links are splits. Red dashed links are merges. Click a row to see its family. Double-click to jump to that object in the Viewer. Zoom with pinch or Ctrl-scroll.

**Family tree.** Two kinds:
- *Future of each object.* Starts at one object and shows what it becomes.
- *Complete family.* Also shows where it came from, and everything that ever merged with it or its relatives. Use this one if you want the full story.

Tick "one overview of every object" to get a single picture of the whole movie. It comes as a PNG and a PDF. Open the PDF if you need to zoom in and read IDs.

## Armadillo / Dia (fixed embryos)

This one is for a specific job: fixed embryos, imaged once, several channels in one `.czi` file.

1. Go to **Armadillo / Dia**.
2. Drop the folder. Subfolders are fine. `.czi` files are read directly, you do not need to convert them.
3. Check the channel numbers. By default Armadillo is channel 2 and Dia is channel 3.
4. Press **Run**.

For each image you get, next to it: `_ARM.tif` and `_DIA.tif` (the two channels, first 20 slices), `_ARM_SEG.tif` (the Armadillo puncta), and the masked images. In the folder you dropped you get one table, `ARM_DIA_Summary.csv`, one row per image.

If you also have cell outlines: put them in a folder called `Cell Outlines` next to the embryo folders, one subfolder per embryo, and press **Run boundary analysis**. It tells you which puncta sit on a cell boundary and which do not.

## Disk space

Be careful here. Tracking needs a lot of room while it works.

Rough rule: **1 GB per timepoint while it runs** for an image of 350 x 250 x 14. A 200 timepoint movie needs about 200 GB free. Bigger images need more, in proportion.

When tracking finishes it deletes what only it needed: its numbered folders (one per timepoint) and the `Features` folder inside `_SegmentationOutput`. What stays is small, a few GB for that same movie: the masks, `TrackedCombined.nii`, the tables and the plots.

The app tells you both numbers before you press Run. Believe them.

Two things follow from this:
- If you run only Segmentation, the `Features` folder stays, because tracking has not used it yet. It is big.
- To track the same image again, run Segmentation again first. It is quick.

If you want to keep the working files, tick "Keep tracking's working files".

## How long it takes

On an Apple M4 laptop, for a 350 x 250 x 14 movie with about 150 objects per timepoint:

- Segmentation: about 2 seconds per timepoint.
- Tracking: about 20 seconds per timepoint.
- Analysis: seconds.

Tracking is the slow part. It does not use the GPU. More objects means more time.

## Train your own model

You need example images where somebody has drawn the objects by hand.

- Each example is one file with two channels, `ZCYX`. Channel 1 is the image. Channel 2 is the mask: object pixels are anything above zero, background is zero.
- Put them all in one folder. All must have the same number of z slices.
- You need at least two files. The last 15 % (by file name) are kept aside to score the model. They are not trained on.

Then:

1. Go to **Training**.
2. Drop the folder.
3. Choose: a new model from scratch, or start from an existing model. Starting from an existing one is much faster if your images look similar.
4. Give it a name. Set the epochs. 100 is normal from scratch.
5. Press **Start training**.

The model shows up in every model list when it finishes. It is saved after every epoch, so if you cancel you keep what you had.

Honest warning: training is slow without an NVIDIA card. On a Mac it is seconds per step. A full training can take a day or more. Use a GPU machine for this.

## Without the window

Everything the app does, you can type. Open a terminal in the ProTrack3D folder.

```bash
conda activate protrack3d-torch
python -m protrack3d download-models
python -m protrack3d models
python -m protrack3d pipeline movie.tif --model BAZOOKA_07212025 --start 1 --end 50
python -m protrack3d batch folder_of_movies --model BAZOOKA_07212025
python -m protrack3d tree movie_TrackingOutput --ids 1-20 --complete --overview
python -m protrack3d train folder_of_examples --name MyModel --epochs 100
```

`python -m protrack3d --help` lists the rest. Add `--help` after any command to see its options.

For a folder run, you can put a `batch_times.txt` in the folder to set the time range per image:

```
all, 1, 50
movie2.tif, 4, 49
```

## Things worth knowing

- **Same input, same result.** Tracking picks some voxels at random when it compares big objects. The random seed is fixed, so two runs give the same answer. Use `--seed -1` if you want it different each time.
- **Smoothed plots.** Curves are averaged over 10 timepoints on each side. Short movies use a shorter window (a sixth of the movie on each side).
- **Edges.** If the image is not a multiple of 32 wide or high, the far edges are mirrored while segmenting and cut off again. Objects right at that edge are a little less reliable.
- **Models from the old TensorFlow version** can be converted. You need the old environment once:

```bash
conda activate protrack3d          # the old TensorFlow environment
python -m protrack3d.convert_tf dump checkpoints/FC-DenseNet/MyOldModel old.npz
conda activate protrack3d-torch
python -m protrack3d import-model old.npz
```

## When it breaks

- **"conda was not found."** Miniforge is not installed, or you installed it somewhere unusual. Install it with the defaults.
- **The first start fails halfway.** Usually the internet dropped. Start it again. It cleans up and retries.
- **No models in the list.** Go to Models and press Download. If the download fails, check your internet and try again.
- **A job says Failed.** Press **Log** on that job, or open the console. The last lines say why. Most often: the disk is full, or the image has more than one channel.
- **Segmentation finds nothing, or finds everything.** Wrong model for this protein. Try another one, or train your own.
- **It is very slow.** Look at the bottom left of the window. If it says CPU, it is not using a GPU. On Windows with an NVIDIA card, update the NVIDIA driver and set it up again: delete the `protrack3d-torch` environment (`conda env remove -n protrack3d-torch`) and start the app.
- **Out of disk in the middle of tracking.** Free space, then run Tracking again on its own. You do not need to redo the segmentation.

Still stuck? Open an issue on this page. Copy the console text into it. That text is what we need.

## License

BSD 3-Clause. See `LICENSE`.
