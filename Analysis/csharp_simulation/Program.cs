using System;
using DSP;
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
            BandpassFilterButterworthImplementation bandpassFilter = new BandpassFilterButterworthImplementation(bottomFrequencyHz:2,topFrequencyHz:30,numSections:3,Fs:100);
            Console.WriteLine("Hello World!");
            int pointCount =2000;
            double[] x = DataGen.Consecutive(pointCount);
            double[] data = new double[2000];
            for(int i =0;i<data.Length;i++){
                data[i] = 
            }

            // var plt = new ScottPlot.Plot(400, 300);
            
            // plt.PlotScatter(x, sin);
            // plt.PlotScatter(x, cos);
            // plt.SaveFig("quickstart.png");
            

        }
    }


}
