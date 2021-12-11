using System;

namespace csfft
{
    public class FFTd
    {
        public delegate void NormalizationFunction(double[] re, double[] im, int n);

        public delegate double[] WindowFunction(int n);

        private int[] bitReverseTable;

        private double[] wnd;

        public int NumOfPoints { get; set; }

        public int Order { get { return (int)Math.Log(NumOfPoints, 2); } }

        public NormalizationFunction NormalizeFunc { get; set; }

        private double[] _or;
        private double[] _oi;

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

        /// <summary>1 / N normalization function.</summary>
        public static void DivByN(double[] re, double[] im, int n)
        {
            if (re.Length < n || im.Length < n)
            {
                throw new RankException("The number of elements is insufficient for the order.");
            }

            for (int i = 0; i < n; i++)
            {
                re[i] /= n;
                im[i] /= n;
            }
        }

        /// <summary>1 / sqrt(N) normalization function.</summary>
        public static void DivBySqrtN(double[] re, double[] im, int n)
        {
            if (re.Length < n || im.Length < n)
            {
                throw new RankException("The number of elements is insufficient for the order.");
            }

            double sqrtn = Math.Sqrt(n);
            for (int i = 0; i < n; i++)
            {
                re[i] /= sqrtn;
                im[i] /= sqrtn;
            }
        }

        /// <summary>Normalization function that does nothing.</summary>
        public static void NoDiv(double[] re, double[] im, int n)
        {
            if (re.Length < n || im.Length < n)
            {
                throw new RankException("The number of elements is insufficient for the order.");
            }
        }

        /// <summary>hann window</summary>
        public static double[] HannWindow(int n)
        {
            double[] wnd = new double[n];
            for (int i = 0; i < n; i++)
            {
                wnd[i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * (1.0 / n) * i);
            }
            return wnd;
        }

        public static double[] RectangularWindow(int n)
        {
            double[] wnd = new double[n];
            for (int i = 0; i < n; i++)
            {
                wnd[i] = 1;
            }
            return wnd;
        }

        public FFTd(int numOfPoints, WindowFunction wndfunc = null, NormalizationFunction normfunc = null)
        {
            if (!IsPow2(numOfPoints))
            {
                throw new ArgumentException("The number of points must be a power of 2.");
            }
            this.NumOfPoints = numOfPoints;

            if (wndfunc == null)
            {
                wndfunc = RectangularWindow;
            }
            wnd = wndfunc(numOfPoints);

            if (normfunc == null)
            {
                normfunc = NoDiv;
            }
            NormalizeFunc = normfunc;

            // Precompute bit reversal subscripts.
            bitReverseTable = new int[numOfPoints];
            for (int i = 1; i < numOfPoints; i++)
            {
                int j = BitReverse(i);
                bitReverseTable[i] = j;
            }

            // Precompute sin and cos.
            int mmax = numOfPoints / 2;
            _or = new double[mmax];
            _oi = new double[mmax];
            for(int i = 0; i < mmax; i++) {
                _or[i] = Math.Cos(-2 * Math.PI / numOfPoints * i);
                _oi[i] = Math.Sin(-2 * Math.PI / numOfPoints * i);
            }
        }

        /// <summary>Forward FFT</summary>
        public void Fwd(double[] srcRe, double[] srcIm, double[] dstRe, double[] dstIm)
        {
            Fwd(srcRe, 0, srcIm, 0, dstRe, dstIm);
        }

        /// <summary>Forward FFT</summary>
        public unsafe void Fwd(double[] srcRe, int reStart, double[] srcIm, int imStart, double[] dstRe, double[] dstIm)
        {
            int pt = NumOfPoints;
            int order = this.Order;

            if (srcRe.Length - reStart < pt || srcIm.Length - imStart < pt)
            {
                throw new RankException("The number of samples is insufficient for the number of points.");
            }

            // apply window function
            for (int i = 0; i < pt; i++)
            {
                dstRe[i] = srcRe[i] * wnd[i];
                dstIm[i] = srcIm[i] * wnd[i];
            }

            // bit-reversal
            for (int i = 1; i < pt; i++)
            {
                int j = bitReverseTable[i];
                if (j > i)
                {
                    (dstRe[j], dstRe[i]) = (dstRe[i], dstRe[j]);
                    (dstIm[j], dstIm[i]) = (dstIm[i], dstIm[j]);
                }
            }

            for (int s = 1; s <= order; s++)
            {
                int m = (int)Math.Pow(2, s);
                int ofactor = pt / m;
                for(int i = 0; i < pt / 2; i++) {
                    int k = (i / (m / 2)) * m;
                    int j = (k == 0) ? i : i % (k / 2);
                    double or = _or[j * ofactor];
                    double oi = _oi[j * ofactor];
                    int kj = k + j;
                    int kjm2 = k + j + m / 2;
                    double tr = or * dstRe[kjm2] - oi * dstIm[kjm2];
                    double ti = or * dstIm[kjm2] + oi * dstRe[kjm2];
                    dstRe[kjm2] = dstRe[kj] - tr;
                    dstIm[kjm2] = dstIm[kj] - ti;
                    dstRe[kj] = dstRe[kj] + tr;
                    dstIm[kj] = dstIm[kj] + ti;
                }
            }

            this.NormalizeFunc(dstRe, dstIm, pt);
        }

        /// <summary>Experimental implementation of forward FFT</summary>
        public (double[], double[]) FwdSimple(double[] srcRe, int reStart, double[] srcIm, int imStart)
        {
            int pt = NumOfPoints;
            int order = this.Order;

            if (srcRe.Length - reStart < pt || srcIm.Length - imStart < pt)
            {
                throw new RankException("The number of samples is insufficient for the number of points.");
            }

            double[] re = new double[pt];
            double[] im = new double[pt];

            // apply window function
            for (int i = 0; i < pt; i++)
            {
                re[i] = srcRe[i] * wnd[i];
                im[i] = srcIm[i] * wnd[i];
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
                for (int k = 0; k < pt; k += m)
                {
                    for (int j = 0; j < m / 2; j++)
                    {
                        double or = Math.Cos(-2 * Math.PI / m * j);
                        double oi = Math.Sin(-2 * Math.PI / m * j);
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

            this.NormalizeFunc(re, im, pt);

            return (re, im);
        }

        /// <summary>Forward FFT for power spectrum</summary>
        public void FwdPower(double[] srcRe, double[] srcIm, double[] tempRe, double[] tempIm, double[] dst, bool db = false, double dbReference = 1)
        {
            FwdPower(srcRe, 0, srcIm, 0, tempRe, tempIm, dst, db, dbReference);
        }


        /// <summary>Forward FFT for power spectrum</summary>
        public void FwdPower(double[] srcRe, int reStart, double[] srcIm, int imStart, double[] tempRe, double[] tempIm, double[] dst, bool db = false, double dbReference = 1)
        {
            int pt = NumOfPoints;
            Fwd(srcRe, reStart, srcIm, imStart, tempRe, tempIm);

            double r2 = dbReference * dbReference;

            if (db)
            {
                for (int i = 0; i < pt; i++)
                {
                    dst[i] = 10 * Math.Log10((tempRe[i] * tempRe[i] + tempIm[i] * tempIm[i]) / r2);
                }
            }
            else
            {
                for (int i = 0; i < pt; i++)
                {
                    dst[i] = (tempRe[i] * tempRe[i] + tempIm[i] * tempIm[i]);
                }
            }
        }
    }
}
