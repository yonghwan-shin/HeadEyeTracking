namespace DSP
{
    public class IIRFilterImplementation
    {
        protected double[] z;
        public IIRFilterImplementation(int order)
        {
            this.z = new double[order];
        }

        public double compute(double input, double[] a)
        {
            // computes y(t) = x(t) + a1*y(t-1) + a2*y(t-2) + ... an*y(t-n)
            // z-transform: H(z) = 1 / (1 - sum(1 to n) [an * y(t-n)])
            // a0 is assumed to be 1
            // y(t) is not stored, so y(t-1) is stored at z[0], 
            // and a1 is stored as coefficient[0]

            double result = input;

            for (int t = 0; t < a.Length; t++)
            {
                result += a[t] * this.z[t];
            }
            for (int t = a.Length - 1; t >= 0; t--)
            {
                if (t > 0)
                {
                    this.z[t] = this.z[t - 1];
                }
                else
                {
                    this.z[t] = result;
                }
            }
            return result;
        }
    }
}