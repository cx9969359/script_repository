import qupath.lib.images.*
import qupath.lib.objects.*
import qupath.lib.objects.classes.PathClass
import qupath.lib.roi.*
import qupath.lib.geom.*

def imageData = getCurrentImageData()
def server = imageData.getServer()

def filename = server.getFile().getName()
def name = filename.take(filename.lastIndexOf('.'))

fileOutPath = "D:\\0528_test\\project\\txt\\" + name + ".txt"
//fileInputPath = "D:/translate_qupath/4-4 V201806300-AGC-FN_2019-01-08 16_32_05_2.txt"

def fileIn = new File(fileInputPath)
def lines = fileIn.readLines()

num_rois = lines.size / 2
print num_rois

for (i = 0; i < num_rois; i++) {
    def label_name = new PathClass(lines[i*2], 0xff8080)
    String[] points = lines[i*2+1].tokenize(';')

    def pointList = [] 
    for (j=0; j< points.length; j++) {
        String point = points[j]
        //print point
        double[] double_point = point.tokenize(',') as double[]
        def a = new Point2(double_point[0], double_point[1])
        pointList << a
        
        //pointList.add(double_point)
    }
    print pointList

    // Create object
    def roiIter = new PolygonROI(pointList)

    def pathObject = new PathAnnotationObject(roiIter, label_name)
    
    // Add object to hierarchy
    addObject(pathObject)
}