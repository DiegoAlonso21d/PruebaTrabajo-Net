using System;
using OpenCvSharp;

class Program
{
    static void Main(string[] args)
    {
        string videoPath = "Videos/video.mp4";
        string processedImagesPath = "ImágenesProcesadas/";
        string facialDifferencesPath = "DiferenciasFaciales/";

        using (var videoCapture = new VideoCapture(videoPath))
        {
            var frameCount = (int)videoCapture.Get(7);

            string fullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Haarcascades", "haarcascade_frontalface_default.xml");
            using (var faceCascade = new CascadeClassifier(fullPath))
            {
                for (int i = 0; i < frameCount; i++)
                {
                    var frame = new Mat();
                    videoCapture.Read(frame);

                    if (frame.Empty())
                        break;

                    var grayFrame = new Mat();
                    Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);

                    var faces = faceCascade.DetectMultiScale(
                        grayFrame,
                        scaleFactor: 1.1,
                        minNeighbors: 3,
                        flags: 0,
                       minSize: new Size(100, 100),  // Ajusta estos valores según sea necesario
                        maxSize: new Size(0, 0)
                      
                    );

                    foreach (var face in faces)
                    {

                         // Calcular el área de la cara
    var faceArea = face.Width * face.Height;

    // Establecer un umbral para la detección de la cara
    if (faceArea < 5000)  // Ajusta este valor según sea necesario
    {
        continue;  // Ignorar detecciones con áreas pequeñas
    }

                        // Dibujar rectángulo alrededor de la cara
                        Cv2.Rectangle(frame, face, Scalar.Red, 2);

                        // Recortar la región de la cara del cuadro original
                        var faceRegion = new Mat(frame, face);

                        // Aplicar ecualización del histograma solo en la región de la cara
                        Cv2.CvtColor(faceRegion, faceRegion, ColorConversionCodes.BGR2GRAY);
                        Cv2.EqualizeHist(faceRegion, faceRegion);

                        // Guardar la región de la cara como una imagen
                        Cv2.ImWrite($"{facialDifferencesPath}cara_{i}.png", faceRegion);
                    }

                    // Guardar el fotograma procesado
                    Cv2.ImWrite($"{processedImagesPath}imagen_{i}.png", frame);
                }
            }
        }
    }
}
