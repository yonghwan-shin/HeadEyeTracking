using System;
using System.Linq;
using MathNet.Filtering;
using MathNet.Filtering.FIR;
using ScottPlot;
namespace csharp_simulation
{
    class Program
    {
//        Demonstrate the use of the filter.
// # First make some data to be filtered.
// T = 5.0         # seconds
// n = int(T * fs) # total number of samples
// t = np.linspace(0, T, n, endpoint=False)
// # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
// data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
        
        static void Main(string[] args)
        {
            var plt = new ScottPlot.Plot(600, 400);
        //signal + noise
            double fs = 1000; //sampling rate
            double fw = 5; //signal frequency
            double fn = 50; //noise frequency
            double n = 5; //number of periods to show
            double A = 10; //signal amplitude
            double N = 1; //noise amplitude
            int size = (int)(n * fs / fw); //sample size

            var t = Enumerable.Range(1, size).Select(p => p * 1 / fs).ToArray();
            var y = t.Select(p => (A * Math.Sin(2 * Math.PI * fw * p)) + (N * Math.Sin(2 * Math.PI * fn * p))).ToArray(); //Original

            //lowpass filter
            double fc = 10; //cutoff frequency
            var lowpass = OnlineFirFilter.CreateLowpass(ImpulseResponse.Finite, fs, fc);

            //bandpass filter
            double fc1 = 0; //low cutoff frequency
            double fc2 = 10; //high cutoff frequency
            var bandpass = OnlineFirFilter.CreateBandpass(ImpulseResponse.Finite, fs, fc1, fc2);

            //narrow bandpass filter
            fc1 = 3; //low cutoff frequency
            fc2 = 7; //high cutoff frequency
            var bandpassnarrow = OnlineFirFilter.CreateBandpass(ImpulseResponse.Finite, fs, fc1, fc2);

            double[] yf1 = lowpass.ProcessSamples(y); //Lowpass
            double[] yf2 = bandpass.ProcessSamples(y); //Bandpass
            double[] yf3 = bandpassnarrow.ProcessSamples(y); //Bandpass Narrow
            plt.PlotSignal(y);
            plt.SaveFig("original.png");
            plt.PlotSignal(yf3);
            plt.SaveFig("test.png");
        }
//     public static double[] Butterworth(double[] indata, double deltaTimeinsec, double CutOff) {
//     if (indata == null) return null;
//     if (CutOff == 0) return indata;

//     double Samplingrate = 1 / deltaTimeinsec;
//     long dF2 = indata.Length - 1;        // The data range is set with dF2
//     double[] Dat2 = new double[dF2 + 4]; // Array with 4 extra points front and back
//     double[] data = indata; // Ptr., changes passed data

//     // Copy indata to Dat2
//     for (long r = 0; r < dF2; r++) {
//         Dat2[2 + r] = indata[r];
//     }
//     Dat2[1] = Dat2[0] = indata[0];
//     Dat2[dF2 + 3] = Dat2[dF2 + 2] = indata[dF2];

//     const double pi = 3.14159265358979;
//     double wc = Math.Tan(CutOff * pi / Samplingrate);
//     double k1 = 1.414213562 * wc; // Sqrt(2) * wc
//     double k2 = wc * wc;
//     double a = k2 / (1 + k1 + k2);
//     double b = 2 * a;
//     double c = a;
//     double k3 = b / k2;
//     double d = -2 * a + k3;
//     double e = 1 - (2 * a) - k3;

//     // RECURSIVE TRIGGERS - ENABLE filter is performed (first, last points constant)
//     double[] DatYt = new double[dF2 + 4];
//     DatYt[1] = DatYt[0] = indata[0];
//     for (long s = 2; s < dF2 + 2; s++) {
//         DatYt[s] = a * Dat2[s] + b * Dat2[s - 1] + c * Dat2[s - 2]
//                    + d * DatYt[s - 1] + e * DatYt[s - 2];
//     }
//     DatYt[dF2 + 3] = DatYt[dF2 + 2] = DatYt[dF2 + 1];

//     // FORWARD filter
//     double[] DatZt = new double[dF2 + 2];
//     DatZt[dF2] = DatYt[dF2 + 2];
//     DatZt[dF2 + 1] = DatYt[dF2 + 3];
//     for (long t = -dF2 + 1; t <= 0; t++) {
//         DatZt[-t] = a * DatYt[-t + 2] + b * DatYt[-t + 3] + c * DatYt[-t + 4]
//                     + d * DatZt[-t + 1] + e * DatZt[-t + 2];
//     }

//     // Calculated points copied for return
//     for (long p = 0; p < dF2; p++) {
//         data[p] = DatZt[p];
//     }

//     return data;
// }
    }


}
