using System;

namespace csfft
{
    public enum NormalizationFlags
    {
        DIV_BY_N,
        DIV_BY_SQRT_N,
        NODIV
    }

    public class FFT
    {
        private int[] bitReverseTable;

        public int NumOfPoints { get; set; }

        public int Order { get { return (int)Math.Log(NumOfPoints, 2); } }

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

        private void normalize(double[] re, double[] im, NormalizationFlags flag)
        {
            if (flag == NormalizationFlags.NODIV)
            {
                return;
            }

            int pt = NumOfPoints;
            double d = 1;

            if (flag == NormalizationFlags.DIV_BY_N)
            {
                d = pt;
            }
            else if (flag == NormalizationFlags.DIV_BY_SQRT_N)
            {
                d = Math.Sqrt(pt);
            }

            for (int i = 0; i < pt; i++)
            {
                re[i] /= d;
                im[i] /= d;
            }
        }

        public FFT(int numOfPoints)
        {
            if (!IsPow2(numOfPoints))
            {
                throw new ArgumentException("The number of points must be a power of 2.");
            }
            this.NumOfPoints = numOfPoints;

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
        public void Fwd(double[] re, double[] im, NormalizationFlags flag = NormalizationFlags.NODIV)
        {
            int pt = NumOfPoints;
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

            normalize(re, im, flag);
        }

        public void FwdSimple(double[] re, double[] im, NormalizationFlags flag = NormalizationFlags.NODIV)
        {
            int pt = NumOfPoints;
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

            normalize(re, im, flag);
        }

        // Fwd FFT in-place
        public double[] FwdPower(double[] re, double[] im,
                 bool db = false,
                 double dbReference = 1,
                 NormalizationFlags flag = NormalizationFlags.NODIV)
        {
            int pt = NumOfPoints;
            Fwd(re, im, NormalizationFlags.NODIV);

            double[] v = new double[pt];
            double r2 = dbReference * dbReference;
            double d = 1;
            if (flag == NormalizationFlags.DIV_BY_N)
            {
                d = pt * pt;
            }
            else if (flag == NormalizationFlags.DIV_BY_SQRT_N)
            {
                d = pt;
            }

            for (int i = 0; i < pt; i++)
            {
                v[i] = (re[i] * re[i] + im[i] * im[i]) / d;
            }

            if (db)
            {
                for (int i = 0; i < pt; i++)
                {
                    v[i] = 10 * Math.Log10(v[i] / r2);
                }
            }

            return v;
        }
    }
}