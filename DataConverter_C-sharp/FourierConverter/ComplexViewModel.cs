using AForge.Math;

namespace FourierConverter
{
    public class ComplexViewModel
    {
        public double Magnitude { get; set; }
        public double Phase { get; set; }
        public double Imaginary { get; set; }
        public double Real { get; set; }

        public ComplexViewModel(Complex complex)
        {
            Magnitude = complex.Magnitude;
            Phase = complex.Phase;
            Real = complex.Re;
            Imaginary = complex.Im;
        }
    }
}