namespace FourierConverter
{
    public class Model
    {
        public string id { get; set; }
        public double[] band_1 { get; set; }
        public double[] band_2 { get; set; }
        public ComplexViewModel[] fourier { get; set; }
        public string inc_angle { get; set; }
        public byte is_iceberg { get; set; }
    }
}