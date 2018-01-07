using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace DataConverter
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var fileStream = File.OpenRead(@"C:\Drive\Projects\StatoilCCoreKaggle\data\test.json"))
            using (TextWriter writer = new StreamWriter(@"C:\Drive\Projects\StatoilCCoreKaggle\data\test.csv"))
            using (StreamReader streamReader = new StreamReader(fileStream))
            using (JsonTextReader reader = new JsonTextReader(streamReader))
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
                            writer.WriteLine("Id,Angle,IsIceberg," + string.Join(",", bandNumbers.Select(b => $"B1_{b}")) + "," + string.Join(",", bandNumbers.Select(b => $"B2_{b}")));
                            hasWrittenHeader = true;
                        }
                        writer.WriteLine($"{c.id}, {c.inc_angle}, {c.is_iceberg}, {string.Join(",", c.band_1)}, {string.Join(",", c.band_2)}");

                        Console.Write($"\rWriting {++j}...");
                    }
                }

            }
        }
    }
}
