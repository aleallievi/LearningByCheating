package cma.examples;
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

import cma.*;

//Command Line Iterations : IterNo
public class CMARoboWalk {
	static PrintWriter pri;
	static String exppath = "";
	static Reader r;
	
    static String paramNames[] = {
	  		  "L3target1",
			  "L4target1",
			  "L5target1",
			  "L3target2",
			  "L5target2"
    };
	static double initParams[] = {-15.0, -30.0, 15.0, 30.0, -30.0};
	//static double initSigma[]={0.01,0.01,1,1,1,1,1,1,1,1,0.005,5.0};
	static double initSigma[]={10,10,10,10,10};

	static int noParams = paramNames.length;
	static int popSize = 0;
	static int totIter = 0;
	public static void writeParams(double pop[][],int curIter) {
         System.out.println("Writing params for iteration " + curIter + "...");
		try {
			for(int i =0;i<pop.length;i++) {
				pri = new PrintWriter(new FileWriter(exppath+"params_"+curIter+"_i_"+i+".txt", true));				
				for (int j = 0; j < noParams; j++) {					
					pri.println(paramNames[j]+"\t"+pop[i][j]);
			
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

	public static void main(String[] args) throws InterruptedException {
	        int curIter = 1;
		exppath = args[0] + "/results/";
		totIter = Integer.parseInt(args[1]);
		boolean fContinue = args[2].equals("-c");
		CMAEvolutionStrategy cma = null;
		if (!fContinue) {
		    popSize = Integer.parseInt(args[2]);
		    cma = new CMAEvolutionStrategy();
		    cma.readProperties(); // read options`, see file CMAEvolutionStrategy.properties
		    cma.setDimension(noParams); // overwrite some loaded properties
		    cma.parameters.setPopulationSize(popSize);
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
		}
		System.out.println(exppath);
		System.out.println(""+totIter);
		double[] fitness;
		if (!fContinue) {
		    fitness = cma.init();
		} else {
		    popSize = cma.parameters.getLambda();
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
