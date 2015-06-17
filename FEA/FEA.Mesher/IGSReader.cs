using System;
using System.Collections.Generic;
using System.IO;

namespace FEA.Mesher
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
			if (StartPt == -1) {
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
					Console.WriteLine (ParameterEntries [ParmCounter]);
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
			ParmCounter = 0;
			int NumEntries = ParameterEntries.Length / 2 + 1;
			ParameterData = new double[NumEntries][];
			for (int iParm = 0; iParm < ParameterEntries.Length; iParm += 2) {
				ParameterData [ParmCounter] = ArrayParser (ParameterEntries [iParm]);
				ParmCounter++;
			}
			Console.Beep ();
		}
		private double[][] ParameterData;

		private double[] ArrayParser(string ArrayString) {
			ArrayString = ArrayString.Remove (ArrayString.Length - 1);
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
						Array.Add (double.Parse (ArrayString));
					}
					break;
				}
				ArrayString = ArrayString.Substring (CharId + 1);
			}
			return Array.ToArray ();
		}
	}
}

