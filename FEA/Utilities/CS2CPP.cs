using System;
using System.Reflection;
using System.Collections.Generic;

namespace Utilities
{
    // writes a c# class to cpp/ cuda code
    public class CS2CPP
    {
        public CS2CPP(List<string> CSharpFiles)
        {
            Typemap = new Dictionary<string, string>();
            // creates a mapping of fundamental data types for C#/C++
            Typemap.Add("System.Int32","int");
            Typemap.Add("System.UInt32", "unsigned int");
            Typemap.Add("System.Single", "float");
            Typemap.Add("System.Double", "double");
            Typemap.Add("System.UInt64", "unsigned long");
            Typemap.Add("System.Int64", "long");
            Typemap.Add("System.Boolean", "bool"); // be careful with this value since byte allignment is crucial in interop
            Typemap.Add("System.Byte", "unsigned char");
            
        }
        // TODO Handle transform a c# generic to a template
        // Make template instatiation dynamic ie (have os compile code at runtime)
        // The purpose of this class is to better suport META programming
        // Memory Layout must be sequential in the C# class
        // compile the files and then use information about the assembly to autogenerate the c++ files
        // Header files must be automatically generated and included (use header generation tool

        Dictionary<string, ClassProperties> ClassNames; // dictionary that holds the index location of the class in the CodeList
        // string is the class name
		Dictionary<string, string> Typemap; // map custom types // maps C# type to c++/c type

    }
    public class ClassProperties {
    	int IndexLocation;
    	List<string> Properties;
    	List<MethodSignature> Methods;
    	bool Templated; // is this
    }
    
    public class MethodSignature {
    	string Name; // name of the method
    	List<string> Arguments; // input/output arguments
    	string ReturnType; // return type of the method
    }
    
}

