using System;
using System.Collections.Generic;

namespace Utilities
{
    // writes a c# class to cpp/ cuda code
    public class CS2CPP
    {
        public CS2CPP(List<List<string>> CSCode)
        {
            
        }
        // TODO Handle transform a c# generic to a template
        // Make template instatiation dynamic ie (have os compile code at runtime)
        // The purpose of this class is to better suport META programming
        // Memory Layout must be sequential in the C# class

        // Header files must be automatically generated and included (use header generation tool

        Dictionary<string, int> ClassNames; // dictionary that holds the index location of the class in the CodeList
        // string is the class name


    }
    public class ClassProperties {
        
    }
}

