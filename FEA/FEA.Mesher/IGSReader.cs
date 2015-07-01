using System;
using System.Collections.Generic;
using System.IO;

namespace FEA.Mesher.IGES
{
	public class IGSReader
	{
		public IGSReader(string Filename)
		{
			var Lines = File.ReadAllLines (Filename);
			int StartPt = -1;
			for (int i = 0; i < Lines.Length; i++) {
				var CurrentLine = Lines [i];
				if (CurrentLine.Substring (71, 2) == "1P") {
					StartPt = i;
					break;
				}
			}
            int DirectoryStartPt = -1;

            if (StartPt == -1) {
                throw new Exception ("Invalid IGES File");
            }

            for (int i = 0; i < StartPt; i++)
            {
                var CurrentLine = Lines[i];
                string KeyPhrase = CurrentLine.Substring(72);
                if (String.Equals(KeyPhrase,"D      1")) {
                    DirectoryStartPt = i;
                    break;
                }
            }    

            if (DirectoryStartPt == -1) {
                throw new Exception ("Invalid IGES File");
            }
			int EndPt = Lines.Length - 2;
			var LastLine = Lines[EndPt];
			LastLine = LastLine.Substring (0, 73);
			int CountStart = LastLine.LastIndexOf (" ") + 1;
			var NumberString = LastLine.Substring (CountStart, 72 - CountStart);
			int ParameterCount = int.Parse (NumberString);
			var ParameterEntries = new string[ParameterCount];
			int ParmCounter = 0;
			ParameterEntries [ParmCounter] = "";
			for (int i = StartPt; i < EndPt + 2; i++) {
				var CurrentLine = Lines [i];
				var ParmEntry = (ParmCounter + 1).ToString () + "P";
				if (! CurrentLine.Contains (ParmEntry)) {
					ParameterEntries [ParmCounter] = ParameterEntries [ParmCounter].Replace (" ", String.Empty);
					ParameterEntries [ParmCounter] = ParameterEntries [ParmCounter].Replace ((ParmCounter + 1).ToString() + "P", String.Empty);
					ParmCounter = ParmCounter + 2;
					if (ParmCounter < ParameterEntries.Length) 	
						ParameterEntries [ParmCounter] = "";
					else {
						break;
					}
				}
				CountStart = CurrentLine.LastIndexOf (" ");
				ParameterEntries [ParmCounter] += CurrentLine.Substring (0, CountStart);
			}

            var DirectoryEntries = new string[StartPt - DirectoryStartPt + 1];
            for (int i = DirectoryStartPt; i < StartPt; i+= 2)
            {
                DirectoryEntries[i - DirectoryStartPt] = Lines[i];
                DirectoryEntries[i - DirectoryStartPt] += Lines[i + 1];
            }

			ParmCounter = 0;
			int NumEntries = ParameterEntries.Length / 2 + 1;
			ParameterData = new double[NumEntries * 2][];
            DirectoryData = new int[NumEntries * 2][];
            Curves = new List<Rational_BSpline_Curve>();
            Surfaces = new List<Rational_BSpline_Surface>();

            for (int iParm = 0; iParm < ParameterEntries.Length; iParm += 2)
            {
                ParameterData[iParm] = ArrayParser (ParameterEntries [iParm]);
                DirectoryData[iParm] = DirectoryStringParser(DirectoryEntries[iParm]);
            }
            for (int iParm = 0; iParm < ParameterEntries.Length; iParm += 2) {
                var Entity = (IGESEntityTypes) (int) (ParameterData[iParm][0]);
                Console.WriteLine(Entity);
                if (Entity == IGESEntityTypes.Rational_BSpline_Curve || Entity == IGESEntityTypes.Rational_BSpline_Surface) {
                    int TransformationMatrixPointer = DirectoryData[iParm][6];
                    int FormNumber = DirectoryData[iParm][14];
                    TransformationMatrix T = null;
                    if (TransformationMatrixPointer > 0)
                    {
                        T = new TransformationMatrix(ParameterData[TransformationMatrixPointer]);
                    }
                    if (Entity == IGESEntityTypes.Rational_BSpline_Curve)
                    {
                        Curves.Add(new IGES.Rational_BSpline_Curve(ParameterData[iParm],T, iParm));
                    }
                    else if (Entity == IGESEntityTypes.Rational_BSpline_Surface)
                    {
                        Surfaces.Add(new IGES.Rational_BSpline_Surface(ParameterData[iParm],T, iParm));
                    }

                }

			}
		}


        public string Filename;


        public List<Rational_BSpline_Curve> Curves;
        public List<Rational_BSpline_Surface> Surfaces;

        private double[][] ParameterData;
        private int[][] DirectoryData;

		private double[] ArrayParser(string ArrayString) {
			ArrayString = ArrayString.Remove (ArrayString.Length - 1);
            ArrayString = ArrayString.Replace("D", "E"); // convert D notation to E
			var Array = new List<double>();
			while (true) {
				if (ArrayString.Length == 0)
					break;
				int CharId = ArrayString.IndexOf (",");
				if (CharId != -1) {
					Array.Add (double.Parse (ArrayString.Substring (0, CharId)));
				} 
				else {
					try {
						Array.Add (double.Parse (ArrayString.Substring (0, ArrayString.Length - 1)));
					} catch (Exception ex) {
                        try {
                            Array.Add (double.Parse (ArrayString));
                        } catch (Exception ex2) {};
					}
					break;
				}
				ArrayString = ArrayString.Substring (CharId + 1);
			}
			return Array.ToArray ();
		}
        private int[] DirectoryStringParser(string DirectoryString) {
            var ans = new int[20];
            for (int i = 0; i < ans.Length; i++)
            {
                string substring = DirectoryString.Substring(i * 8, 8);
                substring = substring.Replace(" ", String.Empty);
                int.TryParse(substring, out ans[i]);
            }
            return ans;
        }
	}

    enum UnitSystem {
        Inches = 1,
        Millimeters = 2,
        //
        Feet = 4,
        Miles = 5,
        Meters = 6,
        Kilometers = 7,
        Mils = 8,
        Microns = 9,
        Centimeters = 10,
        Microinches = 11
    }

}

