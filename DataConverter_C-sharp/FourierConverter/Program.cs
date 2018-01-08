using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using AForge.Imaging;
using AForge.Imaging.Filters;
using AForge.Math;
using Newtonsoft.Json;

namespace FourierConverter
{
    class Program
    {

        [STAThread]
        static void Main(string[] args)
        {
            OpenFileDialog dialog1 = new OpenFileDialog();
            dialog1.Title = "Select file to convert";
            dialog1.InitialDirectory = Environment.CurrentDirectory;
            var result = dialog1.ShowDialog();
            if (result != DialogResult.OK)
                return;
            var f1 = new FileInfo(dialog1.FileName);

            SaveFileDialog dialog2 = new SaveFileDialog(); dialog2.Title = "Select output file placement";
            dialog2.InitialDirectory = f1.Directory.FullName;
            var result2 = dialog2.ShowDialog();
            if (result2 != DialogResult.OK)
                return;
            var f2 = new FileInfo(dialog2.FileName);

            using (var fileStream = File.OpenRead(f1.FullName))
            using (TextWriter writer = new StreamWriter(f2.FullName))
            using (StreamReader streamReader = new StreamReader(fileStream))
            using (JsonTextReader reader = new JsonTextReader(streamReader))
            {
                //WriteToCSV(writer, reader);
                WriteToJson(writer, reader);
            }

            Console.WriteLine("\r\nWriting complete. Press Enter to exit");

            Console.Read();
        }

        private static void WriteToCSV(TextWriter writer, JsonTextReader reader)
        {
            reader.SupportMultipleContent = true;

            var serializer = new JsonSerializer();
            bool hasWrittenHeader = false;

            int j = 0;
            while (reader.Read())
            {
                if (reader.TokenType == JsonToken.StartObject)
                {
                    Model c = serializer.Deserialize<Model>(reader);

                    if (!hasWrittenHeader)
                    {
                        int i = 1;
                        var bandNumbers = c.band_1.Select(b => i++).ToArray();
                        writer.WriteLine("Id,Angle,IsIceberg," + string.Join(",", bandNumbers.Select(b => $"B1_{b}")) + "," + string.Join(",", bandNumbers.Select(b => $"B2_{b}")) + ',' + string.Join(",", bandNumbers.Select(b => $"F_Re{b},F_Im{b},F_Ma{b},F_Ph{b}")));

                        hasWrittenHeader = true;
                    }
                    ApplyFourier(c);
                    writer.WriteLine($"{c.id}, {c.inc_angle}, {c.is_iceberg}, {string.Join(",", c.band_1)}, {string.Join(",", c.band_2)}, {string.Join(",", c.fourier.Select(f => $"{f.Real}, {f.Imaginary}, {f.Magnitude}, {f.Phase}"))}");

                    Console.Write($"\rWriting {++j}... {minBand} / {maxBand}");
                }
            }
        }

        private static void WriteToJson(TextWriter writer, JsonTextReader reader)
        {
            reader.SupportMultipleContent = true;

            var serializer = new JsonSerializer();
            writer.Write('[');

            int j = 0;
            while (reader.Read())
            {
                if (reader.TokenType == JsonToken.StartObject)
                {
                    Model c = serializer.Deserialize<Model>(reader);

                    ApplyFourier(c);
                    serializer.Serialize(writer, c);

                    Console.Write($"\rWriting {++j}... {minBand} / {maxBand}");
                }
            }
            writer.Write(']');
        }

        private static double minBand;
        private static double maxBand;

        private static void ApplyFourier(Model model)
        {
            var map1 = new Bitmap(75, 75);
            var map2 = new Bitmap(75, 75);

            for (int i = 0; i < 75; i++)
            {
                for (int j = 0; j < 75; j++)
                {
                    var band1Value = model.band_1[i * 75 + j];
                    var band2Value = model.band_2[i * 75 + j];

                    minBand = Math.Min(minBand, band1Value);
                    minBand = Math.Min(minBand, band2Value);
                    maxBand = Math.Max(maxBand, band1Value);
                    maxBand = Math.Max(maxBand, band2Value);

                    var pixel1Value = Math.Max(0, Math.Min(255, ((int) band1Value + 50)*3));
                    map1.SetPixel(i,j, Color.FromArgb(pixel1Value,pixel1Value,pixel1Value));

                    var pixel2Value = Math.Max(0, Math.Min(255, ((int)band2Value + 50)*3));
                    map2.SetPixel(i, j, Color.FromArgb(pixel2Value, pixel2Value, pixel2Value));
                }
            }

            map1 = cropImage(map1, new Rectangle(5, 5, 64, 64));
            var bmp8bpp = Grayscale.CommonAlgorithms.BT709.Apply(map1);
            ComplexImage complexImage = ComplexImage.FromBitmap(bmp8bpp);
            complexImage.ForwardFourierTransform();

            model.fourier = new ComplexViewModel[map1.Width * map1.Height];
            int k = 0;
            foreach (var complex in complexImage.Data)
            {
                model.fourier[k] = new ComplexViewModel(complex);
                k++;
            }
        }

        public static Bitmap cropImage(Bitmap b, Rectangle r)
        {
            Bitmap nb = new Bitmap(r.Width, r.Height);
            Graphics g = Graphics.FromImage(nb);
            g.DrawImage(b, -r.X, -r.Y);
            return nb;
        }
    }
}
