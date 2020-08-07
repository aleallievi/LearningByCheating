package cma;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.StreamTokenizer;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;
import java.util.Vector;
import java.util.HashMap;

import cma.*;

//Command Line Iterations : IterNo
public class CMAMain {
    static PrintWriter pri;
    static String exppath = "";
    static Reader r;
    
    static Vector<String> paramNames = new Vector<String>();
    static HashMap<String,Double> initParamsMap = new HashMap<String,Double>();
    static HashMap<String,Double> initSigmaMap = new HashMap<String,Double>();

    static int popSize = 0;
    static int totIter = 0;
    public static void writeParams(double pop[][],int curIter) {
         System.out.println("Writing params for iteration " + curIter + "...");
	 try {
	     for(int i =0;i<pop.length;i++) {
		 pri = new PrintWriter(new FileWriter(exppath+"params_"+curIter+"_i_"+i+".txt", true));				
		 for (int j = 0; j < paramNames.size(); j++) {					
		     pri.println(paramNames.get(j)+"\t"+pop[i][j]);
		     
		 }
		 pri.close();
	     }				
	     pri = new PrintWriter(new FileWriter(exppath+"paramswritten_"+curIter+".txt", true));				
	     //pri.println("\n");
	     pri.close();
	 }
	 catch (Exception e) {
	     e.printStackTrace();
	     System.out.println("No such file exists.");
	 }							
	 
	 
    }
    
    public static double[] readFitness(int curIter) {
	double fitness[] = new double[popSize];
	try {			
	    for(int i =0;i<popSize;i++) {
		String expectedFile = exppath+"value_"+curIter+"_i_"+i+".txt";
		if (! new File(expectedFile).exists()) {
		    fitness[i] = 100000; // Absurdly high value, taken as +INFINITY
		    System.out.println("Could not find file " + expectedFile + " so we'll assume bad fitness");
		} else { 
		    r = new BufferedReader(new FileReader(expectedFile));
		    StreamTokenizer stok = new StreamTokenizer(r);
		    stok.parseNumbers();
		    stok.nextToken();				    
		    while (stok.ttype != StreamTokenizer.TT_EOF) {
			if (stok.ttype == StreamTokenizer.TT_NUMBER){
			    fitness[i]=-1*stok.nval;
			}			          
			stok.nextToken();
		    }
		    r.close();
		}
	    }
	}
	
	catch (Exception e) {
	    e.printStackTrace();
	    System.out.println("No such file exists.");
	}
	return fitness;
    }
    
    public static boolean initializeValuesFromInputFile(String inputFilename) {
	//StringBuilder sb = new StringBuilder();
	try {
	    FileReader fr = new FileReader(inputFilename);
	    BufferedReader br = new BufferedReader(fr);

	    StreamTokenizer st = new StreamTokenizer(br);
	    st.slashSlashComments(true);
	    st.slashStarComments(true);
	    st.parseNumbers();
	    st.commentChar('#');
	    st.wordChars('_', '_');
	    
	    int lastLine = -1;
	    String paramName = null;
	   
	    while (st.nextToken() != st.TT_EOF) {
		if (lastLine != st.lineno()) {
		    lastLine = st.lineno();
		    paramName = st.sval;
		    if (!paramNames.contains(paramName)) {
			paramNames.add(paramName);
		    }
		} else {
		    if (st.ttype != st.TT_NUMBER) {
			System.err.println("Error: Bad numerical value " + st.toString()); 
			return false;
		    }
		    Double value = Double.valueOf(st.nval);
		    if (!initParamsMap.containsKey(paramName)) {
			initParamsMap.put(paramName, value);
		    } else if (!initSigmaMap.containsKey(paramName)) {
			initSigmaMap.put(paramName, value);
		    } else {
			System.err.println("Error: extra value " + st.toString() + " for parameter " + paramName); 
			return false;
		    }
		    
		}
		
	    }
	    br.close();
	}
	catch (IOException e) {
	    System.err.println("Error: " + e);
	    return false;
	}
	
	boolean valid = true;
	for (int i = 0; i < paramNames.size(); i++) {
	    String paramName = paramNames.get(i);
	    if (!initParamsMap.containsKey(paramName)) {
		System.err.println("Error: No initial value for parameter " + paramName);
		 valid = false;
	    }
	    if (!initSigmaMap.containsKey(paramName)) {
		System.err.println("Error: No initial sigma for parameter " + paramName);
		 valid = false;
	    }
	}

	return valid;	
    }
    
    public static void printUsage() {
	System.err.println("Usage: java -cp java cma.CMAMain <experiment_path> <num_iter> <pop_size> <input_file> [--seed <seed>]");
	System.err.println("       java -cp java cma.CMAMain <experiment_path> <num_iter> -c <iter_to_contine_from>");
    }

    public static void main(String[] args) throws InterruptedException {
	int curIter = 1;
	if (args.length < 4) {
	    printUsage();
	    return;
	}
	exppath = args[0] + "/results/";
	totIter = Integer.parseInt(args[1]);
	boolean fContinue = args[2].equals("-c");

	CMAEvolutionStrategy cma = null;
	if (!fContinue) {
	    popSize = Integer.parseInt(args[2]);
	    String inputFilename = args[3];
            long seed = -1;
            if (args.length >= 6) {
                if (args[4].equals("--seed"))
                    seed = Long.parseLong(args[5]);
            }
	    if (!initializeValuesFromInputFile(inputFilename)) {
		System.err.println("Error in parsing input file " + inputFilename);
		return;
	    } 
	    System.out.println("Successfully Parsed " + inputFilename);

	    double[] initParams = new double[paramNames.size()];
	    double[] initSigma = new double[paramNames.size()];
	    for (int i = 0; i < paramNames.size(); i++) {
		initParams[i] = initParamsMap.get(paramNames.get(i)).doubleValue();
		initSigma[i] = initSigmaMap.get(paramNames.get(i)).doubleValue();
	    }
	    
	    cma = new CMAEvolutionStrategy();
	    cma.readProperties(); // read options`, see file CMAEvolutionStrategy.properties
	    cma.setDimension(paramNames.size()); // overwrite some loaded properties
	    cma.parameters.setPopulationSize(popSize);
            cma.setSeed(seed);
	    cma.parameters.setNames(paramNames);
	    cma.setInitialX(initParams);
	    cma.setInitialStandardDeviations(initSigma); // also a mandatory setting
	} else {
	    curIter = Integer.parseInt(args[3]);
	    try {
		FileInputStream fileIn = new FileInputStream(args[0] + "/process/cma.ser");
		ObjectInputStream in = new ObjectInputStream(fileIn);
		cma = (CMAEvolutionStrategy) in.readObject();
		in.close();
		fileIn.close();
	    } catch(IOException i) {
		i.printStackTrace();
		return;
	    } catch(ClassNotFoundException c) {
		System.out.println("CMAEvolutionStrategy class not found.");
		c.printStackTrace();
		return;
	    }
	    popSize = cma.parameters.getLambda();
	    paramNames = cma.parameters.getNames();
	}
	System.out.println(exppath);
	System.out.println(""+totIter);
	double[] fitness;
	if (!fContinue) {
	    fitness = cma.init();
	} 
	for (; curIter <= totIter ; curIter++) {
	    if (!fContinue) {
		double[][] pop = cma.samplePopulation(); // get a new population of solutions	    	
		writeParams(pop,curIter);
		try {
		    FileOutputStream fileOut = new FileOutputStream(args[0] + "/process/cma.ser");
		    ObjectOutputStream out = new ObjectOutputStream(fileOut);
		    out.writeObject(cma);
		    out.close();
		    fileOut.close();
		} catch(IOException i) {
		    i.printStackTrace();
		}
	    }
	    fContinue = false;
	    
	    if (curIter == totIter) { 
		System.out.println("CMA-ES done.  The last iteration should be evaluating now.");
		break; 
	    }
	    String expectedFile = exppath + "valuationdone_" + curIter + ".txt";
	    
	    File f = new File(expectedFile);
	    while(true) {
		//wait
		if(f.exists()) {
		    System.out.println("Found " + expectedFile + "; Generating next parameter set\n");
		    break;
		} else {
		    Thread.sleep(5000);
		    System.out.println("Waiting on " + expectedFile);
		}
	    }
	    //Read fitness function
	    fitness = readFitness(curIter);
	    cma.updateDistribution(fitness);        // pass fitness array to update search distribution
	}
    } // main  
} // class
