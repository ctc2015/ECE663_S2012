package cetus.exec;

import cetus.analysis.*;
import cetus.codegen.CodeGenPass;
import cetus.codegen.ompGen;
import cetus.hir.PrintTools;
import cetus.hir.Program;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.transforms.*;

import java.io.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.ArrayList;

public class LoopSkewDriver extends Driver {
	// ...
	
	protected LoopSkewDriver() {
		super();
	}	
	public static void main(String[] args) {
		(new LoopSkewDriver()).run(args);
	}
	

	public void run(String[] args) {
		parseCommandLine(args);

		parseFiles();

		if (getOptionValue("parse-only") != null) {
			System.err.println("parsing finished and parse-only option set");
			Tools.exit(0);
		}

//		runPasses();
//		Run loop skewing pass
//		System.out.println("Running Loop Skew pass with this Driver");
		TransformPass.run(new LoopSkewPass(program));
		
		PrintTools.printlnStatus("Printing...", 1);

		try {
			program.print();
		} catch (IOException e) {
			System.err.println("could not write output files: " + e);
			Tools.exit(1);
		}
	}
	
}
