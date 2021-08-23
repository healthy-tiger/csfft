using System;

namespace csfft
{
    public class FFT
    {
        private int[] bitReverseTable;

        private int _pt;
        public int NumOfPoints { get { return _pt; } }

        public int Order { get { return (int)Math.Log(_pt, 2); } }

        private double[][] omegaRe;
        private double[][] omegaIm;

        public static bool IsPow2(int x)
        {
            return (x & (x - 1)) == 0;
        }

        private int BitReverse(int v)
        {
            int maxBits = this.Order;
            int r = 0;
            int nbits = maxBits - 1;

            for (int i = 0; i < maxBits; i++)
            {
                r |= (1 << (nbits - i)) * (((1 << i) & v) >> i);
            }

            return r;
        }

        public FFT(int numOfPoints)
        {
            if (!IsPow2(numOfPoints))
            {
                throw new ArgumentException("The number of points must be a power of 2.");
            }
            _pt = numOfPoints;

            // Precompute bit reversal subscripts.
            bitReverseTable = new int[numOfPoints];
            for (int i = 1; i < numOfPoints; i++)
            {
                int j = BitReverse(i);
                bitReverseTable[i] = j;
            }

            // Precompute omega.
            int order = this.Order;
            omegaRe = new double[order][];
            omegaIm = new double[order][];
            for (int s = 1; s <= order; s++)
            {
                int m = (int)Math.Pow(2, s);

                omegaRe[s - 1] = new double[m / 2];
                omegaIm[s - 1] = new double[m / 2];

                for (int j = 0; j < m / 2; j++)
                {
                    omegaRe[s - 1][j] = Math.Cos(-2 * Math.PI / m * j);
                    omegaIm[s - 1][j] = Math.Sin(-2 * Math.PI / m * j);
                }
            }
        }

        // Fwd FFT in-place
        public void Fwd(double[] re, double[] im)
        {
            int pt = _pt;
            int order = this.Order;

            if (re.Length < pt || im.Length < pt)
            {
                throw new ArgumentException("The number of samples is insufficient for the number of points.");
            }

            // bit-reversal
            for (int i = 1; i < pt; i++)
            {
                int j = bitReverseTable[i];
                if (j > i)
                {
                    (re[j], re[i]) = (re[i], re[j]);
                    (im[j], im[i]) = (im[i], im[j]);
                }
            }

            for (int s = 1; s <= order; s++)
            {
                int m = (int)Math.Pow(2, s);
                double[] omr = omegaRe[s - 1];
                double[] omi = omegaIm[s - 1];
                for (int k = 0; k < pt; k += m)
                {
                    for (int j = 0; j < m / 2; j++)
                    {
                        double or = omr[j];
                        double oi = omi[j];
                        int kj = k + j;
                        int kjm2 = k + j + m / 2;
                        (double tr, double ti) = ((or * re[kjm2]) - (oi * im[kjm2]),
                                      (or * im[kjm2]) + (oi * re[kjm2]));
                        (double ur, double ui) = (re[kj],
                                      im[kj]);
                        (re[kj], im[kj]) = (ur + tr,
                                    ui + ti);
                        (re[kjm2], im[kjm2]) = (ur - tr,
                                    ui - ti);
                    }
                }
            }
        }

        public void FwdSimple(double[] re, double[] im)
        {
            int pt = _pt;
            int order = this.Order;

            if (re.Length < pt || im.Length < pt)
            {
                throw new ArgumentException("The number of samples is insufficient for the number of points.");
            }

            // bit-reversal
            for (int i = 1; i < pt; i++)
            {
                int j = BitReverse(i);
                if (j > i)
                {
                    (re[j], re[i]) = (re[i], re[j]);
                    (im[j], im[i]) = (im[i], im[j]);
                }
            }

            for (int s = 1; s <= order; s++)
            {
                int m = (int)Math.Pow(2, s);
                double omr = Math.Cos(-2 * Math.PI / m);
                double omi = Math.Sin(-2 * Math.PI / m);
                for (int k = 0; k < pt; k += m)
                {
                    double or = 1; // cos(0)
                    double oi = 0; // sin(0)
                    for (int j = 0; j < m / 2; j++)
                    {
                        int kj = k + j;
                        int kjm2 = k + j + m / 2;
                        (double tr, double ti) = ((or * re[kjm2]) - (oi * im[kjm2]),
                                      (or * im[kjm2]) + (oi * re[kjm2]));
                        (double ur, double ui) = (re[kj],
                                      im[kj]);
                        (re[kj], im[kj]) = (ur + tr,
                                    ui + ti);
                        (re[kjm2], im[kjm2]) = (ur - tr,
                                    ui - ti);

                        (or, oi) = (or * omr - oi * omi,
                                or * omi + oi * omr); // Rotate by -2 * PI / m.
                    }
                }
            }
        }
    }
}
