using System;
using System.Numerics;

namespace csfft
{

    public class FFTv
    {
        public delegate void NormalizationFunction(Vector2[] s, int n);

        public delegate float[] WindowFunction(int n);

        private int[] bitReverseTable;

        private float[] wnd;

        public int NumOfPoints { get; set; }

        public int Order { get { return (int)Math.Log(NumOfPoints, 2); } }

        public NormalizationFunction NormalizeFunc { get; set; }

        private Vector2[] _o;

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
        public static void DivByN(Vector2[] s, int n)
        {
            if (s.Length < n)
            {
                throw new RankException("The number of elements is insufficient for the order.");
            }

            for (int i = 0; i < n; i++)
            {
                s[i] /= n;
            }
        }

        /// <summary>1 / sqrt(N) normalization function.</summary>
        public static void DivBySqrtN(Vector2[] s, int n)
        {
            if (s.Length < n)
            {
                throw new RankException("The number of elements is insufficient for the order.");
            }

            float sqrtn = (float)Math.Sqrt(n);
            for (int i = 0; i < n; i++)
            {
                s[i] /= sqrtn;
            }
        }

        /// <summary>Normalization function that does nothing.</summary>
        public static void NoDiv(Vector2[] s, int n)
        {
            if (s.Length < n)
            {
                throw new RankException("The number of elements is insufficient for the order.");
            }
        }

        /// <summary>hann window</summary>
        public static float[] HannWindow(int n)
        {
            float[] wnd = new float[n];
            for (int i = 0; i < n; i++)
            {
                wnd[i] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * (1.0 / n) * i));
            }
            return wnd;
        }

        public static float[] RectangularWindow(int n)
        {
            float[] wnd = new float[n];
            for (int i = 0; i < n; i++)
            {
                wnd[i] = 1;
            }
            return wnd;
        }

        public FFTv(int numOfPoints, WindowFunction wndfunc = null, NormalizationFunction normfunc = null)
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
            _o = new Vector2[mmax];
            for (int i = 0; i < mmax; i++)
            {
                _o[i].X = (float)Math.Cos(-2 * Math.PI / numOfPoints * i);
                _o[i].Y = (float)Math.Sin(-2 * Math.PI / numOfPoints * i);
            }
        }

        /// <summary>Forward FFT</summary>
        public void Fwd(Vector2[] src, Vector2[] dst)
        {
            Fwd(src, 0, dst);
        }

        /// <summary>Forward FFT</summary>
        public unsafe void Fwd(Vector2[] src, int srcStart, Vector2[] dst)
        {
            int pt = NumOfPoints;
            int order = this.Order;

            if (src.Length - srcStart < pt || dst.Length < pt)
            {
                throw new RankException("The number of samples is insufficient for the number of points.");
            }

            // apply window function
            for (int i = 0; i < pt; i++)
            {
                dst[i] = src[i] * wnd[i];
            }

            // bit-reversal
            for (int i = 1; i < pt; i++)
            {
                int j = bitReverseTable[i];
                if (j > i)
                {
                    (dst[j], dst[i]) = (dst[i], dst[j]);
                }
            }

            fixed (Vector2* pdst = &dst[0], po = &_o[0])
            {
                for (int s = 1; s <= order; s++)
                {
                    int m = (int)Math.Pow(2, s);
                    int ofactor = pt / m;
                    for (int k = 0; k < pt; k += m)
                    {
                        for (int j = 0; j < m / 2; j++)
                        {
                            Vector2* o = po + j * ofactor;
                            Vector2* pkj = pdst + k + j;
                            Vector2* pkjm2 = pkj + m / 2;
                            Vector2 t;
                            t.X = o->X * pkjm2->X - o->Y * pkjm2->Y;
                            t.Y = o->X * pkjm2->Y + o->Y * pkjm2->X;
                            *pkjm2 = *pkj - t;
                            *pkj = *pkj + t;
                        }
                    }
                }
            }

            this.NormalizeFunc(dst, pt);
        }

        /// <summary>Forward FFT for power spectrum</summary>
        public void FwdPower(Vector2[] src, Vector2[] temp, float[] dst, bool db = false, float dbReference = 1)
        {
            FwdPower(src, 0, temp, dst, db, dbReference);
        }


        /// <summary>Forward FFT for power spectrum</summary>
        public void FwdPower(Vector2[] src, int start, Vector2[] temp, float[] dst, bool db = false, float dbReference = 1)
        {
            int pt = NumOfPoints;
            Fwd(src, start, temp);

            float r2 = dbReference * dbReference;

            if (db)
            {
                for (int i = 0; i < pt; i++)
                {
                    dst[i] = (float)(10 * Math.Log10(temp[i].LengthSquared()) / r2);
                }
            }
            else
            {
                for (int i = 0; i < pt; i++)
                {
                    dst[i] = temp[i].LengthSquared();
                }
            }
        }
    }
}
