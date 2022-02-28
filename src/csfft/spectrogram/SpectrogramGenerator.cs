using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Numerics;

namespace csfft.spectrogram
{
    public class SpectrogramGenerator
    {
        /**
         * <summary>number of fft points</summary>
         */
        public int FftSize { get { return trans.NumOfPoints; } }

        private FFTv trans;

        private Vector2[] _src;

        private Vector2[] _temp;

        private List<float[]> _powers;

        /**
         * <summary>The highest level represented in the spectrogram</summary>
         * <remarks>All levels above "gain" are represented by the hue specified by "HueMax".</remarks>
         */
        public int Gain { get; set; }

        /**
         * <summary>Range below "gain" expressed in spectrogram</summary>
         * <remarks>The range from "gain" to "gain"-"range" is expressed by the hue from "HueMax" to "HueMin".</remarks>
         */
        public int Range { get; set; }

        /**
         * <summary>Hue corresponding to the highest level</summary>
         */
        public double HueMax { get; set; }

        /**
         * <summary>Hue corresponding to the lowest level</summary>
         */
        public double HueMin { get; set; }

        public SpectrogramGenerator(int fftsize, FFTv.WindowFunction windowFunction = null, int Gain = -30, int Range = 80, double HueMax = 0, double HueMin = (double)2 / 3)
        {
            if (windowFunction == null)
            {
                windowFunction = FFTv.HannWindow;
            }
            this.trans = new FFTv(fftsize, windowFunction, FFTv.DivByN);
            this.Gain = Gain;
            this.Range = Range;
            this.HueMax = HueMax;
            this.HueMin = HueMin;
            Reset();
        }

        public void Reset()
        {
            this._src = new Vector2[FftSize];
            this._temp = new Vector2[FftSize];
            this._powers = new List<float[]>();
        }

        public void PutNextFrame(Vector2[] src, int offset, int length)
        {
            int d = FftSize - length;
            if (d > 0)
            {
                Array.Copy(_src, length, _src, 0, d);
                Array.Copy(src, offset, _src, d, length);
            }
            else
            {
                Array.Copy(src, offset, _src, 0, FftSize);
            }

            float[] dst = new float[FftSize];
            trans.FwdPower(_src, _temp, dst, db: true);
            _powers.Add(dst);
        }

        public void PutNextFrame(float[] src, int offset, int length)
        {
            int d = FftSize - length;
            if (d > 0)
            {
                Array.Copy(_src, length, _src, 0, d);
                for (int i = 0; i < length; i++)
                {
                    _src[i + d].X = src[i + offset];
                }
            }
            else
            {
                for (int i = 0; i < FftSize; i++)
                {
                    _src[i].X = src[i + offset];
                }
            }

            float[] dst = new float[FftSize];
            trans.FwdPower(_src, _temp, dst, db: true);
            _powers.Add(dst);
        }


        public void PutAll(Vector2[] src, int length, int slide, int offset = 0)
        {
            if (slide <= 0)
            {
                throw new ArgumentException("Slide widths less than 0 cannot be accepted.");
            }
            for (int off = 0; off < length; off += slide)
            {
                int len = Math.Min(length - off, slide);
                this.PutNextFrame(src, off + offset, len);
            }
        }

        public void PutAll(float[] src, int length, int slide, int offset = 0)
        {
            if (slide <= 0)
            {
                throw new ArgumentException("Slide widths less than 0 cannot be accepted.");
            }
            for (int off = 0; off < length; off += slide)
            {
                int len = Math.Min(length - off, slide);
                this.PutNextFrame(src, off + offset, len);
            }
        }

        public const int PixelWidth = 3;

        public (int width, int height) GetSpectrogramSize()
        {
            return (_powers.Count, _powers[0].Length / 2);
        }

        public void DrawSpectrogram(IntPtr pixbuffer, int stride, bool clear = false)
        {
            float[][] power = _powers.ToArray();
            (int width, int height) = GetSpectrogramSize();
            double hmin = HueMin;
            double hmax = HueMax;

            byte[] bytes = new byte[stride * height];
            for (int col = 0; col < width; col++)
            {
                for (int row = 0; row < height; row++)
                {
                    int off = col * PixelWidth + (height - row - 1) * stride;
                    double m = power[col][row]; // Since the input is normalized to the range -1.0 to 1.0, the dB of the FFT result will always be 0 or minus.
                    ColorUtils.dbToColor(m, Gain, Range, hmin, hmax, bytes, off);
                }
            }
            Marshal.Copy(bytes, 0, pixbuffer, bytes.Length);

            if (clear)
            {
                _powers.Clear();
            }
        }
    }
}