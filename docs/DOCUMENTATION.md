# Documentation

Put training images into a folder

    ffmpeg -framerate 2 -pattern_type glob -i 'm8_test_view_rendered_*.png'  -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le output.mov

    ffmpeg -i output.mov -vf "fps=10,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" output.gif