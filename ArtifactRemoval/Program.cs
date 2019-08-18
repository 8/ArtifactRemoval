using System;
using System.IO;
using System.Linq;
using OpenCvSharp;

namespace ArtifactRemoval
{
  class Program
  {
    static void Main(string[] args)
    {
      new Program().Run(args[0]);
    }

    void Run(string file)
    {
      var input = File.ReadAllBytes(file);
      using (var mat = Mat.FromImageData(input))
      using (var grey = mat.CvtColor(ColorConversionCodes.BGR2GRAY))
      using (var thresholded = new Mat())
      {
        Cv2.AdaptiveThreshold(grey, thresholded, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv, 15, 7);
        thresholded.SaveImage(GetModifiedName(file, "thresholded"));

        /* close the gaps between the artifacts by using dilation */
        int dilateX = 15, dilateY = 3;
        using (var dilateKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(dilateX, dilateY)))
        using (var morphed = new Mat())
        {
          Cv2.MorphologyEx(thresholded, morphed, MorphTypes.Dilate, dilateKernel);
          morphed.SaveImage(GetModifiedName(file, "morphed"));

          /* find external contours */
          Mat[] contours; var hierarchy = new Mat();
          Cv2.FindContours(morphed, out contours, hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

          /* draw contours */
          using (var contourImage = new Mat())
          {
            Cv2.CvtColor(grey, contourImage, ColorConversionCodes.GRAY2BGR);
            Cv2.DrawContours(contourImage, contours, -1, new Scalar(0, 0, 255), 1);
            contourImage.SaveImage(GetModifiedName(file, "contour"));
          }

          var imageHeight = mat.Height;
          var yThreshold = 2;

          var items = contours
            .Select(c => new { Contour = c, BoundingBox = c.BoundingRect() })
            .ToArray();

          /* filter all contours that are near the top border or bottom border */
          var filteredTopBottom = items
            .Where(i => i.BoundingBox.Top <= yThreshold ||
                        (imageHeight - i.BoundingBox.Bottom) <= yThreshold)
            .ToArray();

          var filteredContours = filteredTopBottom.Select(i => i.Contour).ToArray();

          DrawContours(mat.Width, imageHeight, GetModifiedName(file, "filtered-top-bottom-contours"),
            filteredContours);

          using (var fixedImage = grey.Clone())
          {
            Cv2.DrawContours(fixedImage, filteredContours, -1, new Scalar(255, 255, 255), Cv2.FILLED);
            fixedImage.SaveImage(GetModifiedName(file, "fixed"));
          }
        }
      }
    }

    void DrawContours(int width, int height, string file, Mat[] contours)
    {
      using (var image = new Mat(new Size(width, height), MatType.CV_8UC3))
      {
        Cv2.DrawContours(image, contours, -1, new Scalar(0, 0, 255), 3);
        image.SaveImage(file);
      }
    }

    string GetModifiedName(string file, string title)
      => $"{Path.GetFileNameWithoutExtension(file)}-{title}.png";

  }
}
