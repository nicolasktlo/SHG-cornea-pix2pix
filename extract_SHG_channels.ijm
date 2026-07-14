inputDir = getDirectory("Choose input folder (CZI files)");
outputDir = getDirectory("Choose output folder");

list = getFileList(inputDir);
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".czi")) {
        run("Bio-Formats Importer", "open=[" + inputDir + list[i] + "] autoscale color_mode=Composite open_all_series=false view=Hyperstack stack_order=XYCZT");
        title = getTitle();
        
        // Extract channel 2 (backwards)
        run("Duplicate...", "duplicate channels=2");
        saveAs("Tiff", outputDir + replace(title, ".czi", "_ch2_backwards.tif"));
        close();
        
        // Extract channel 5 (forwards)
        selectWindow(title);
        run("Duplicate...", "duplicate channels=5");
        saveAs("Tiff", outputDir + replace(title, ".czi", "_ch5_forwards.tif"));
        close();
        
        selectWindow(title);
        close();
    }
}
print("Done!");