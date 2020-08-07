package cma.examples;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.StreamTokenizer;

import cma.*;

//Command Line Iterations : IterNo
public class GenerateParams {
	static PrintWriter pri;
	static String exppath = "/home/suyog/cmaes/temp/";
	static Reader r;
	
	public static void writeParams(double pop[][],double sigma[],int curIter,int noParams)
	{
		try
		{
			for(int i =0;i<pop.length;i++)
			{
				pri = new PrintWriter(new FileWriter(exppath+"params_"+curIter+"_"+i+".txt", true));				
				for (int j = 0; j < noParams; j++)
				{
					System.out.println(j);
					pri.println(pop[i][j]);
			
				}
				pri.close();
			}
			pri = new PrintWriter(new FileWriter(exppath+"sigma_"+curIter+".txt", true));
			for (int j = 0; j < sigma.length; j++)
			{
				pri.println(sigma[j]);
		
			}
		
			pri.close();				
			
		}
		catch (Exception e)
		{
			e.printStackTrace();
			System.out.println("No such file exists.");
		}							

	
	}
	
	public static void main(String[] args) {
		
		int curIter = Integer.parseInt(args[0]);
		System.out.println(curIter);
		if(curIter ==1)
		{			
			CMAEvolutionStrategy cma = new CMAEvolutionStrategy();
			cma.readProperties(); // read options`, see file CMAEvolutionStrategy.properties
			cma.setDimension(11); // overwrite some loaded properties
			double initX[]={0.1,-0.05,-4,1.2,7,8,-99,-0.6,0.8,3.4,1.5};
			double initSigma[]={0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2};
			System.out.println(initSigma.length);
			cma.setInitialStandardDeviations(initSigma); // also a mandatory setting
			double[][] pop = cma.samplePopulation(); // get a new population of solutions
			writeParams(pop,initSigma,curIter,initX.length);			
			
		}
		
		else
		{
			int prevIter = curIter-1;
			double fitness[] = new double[11];
			double prevSigma[]=new double[11];
			double prevPop[][]=new double[11][11];
			try
			{
				
				int count=0;
				for(int i =0;i<11;i++)
					{
					r = new BufferedReader(new FileReader(exppath+"value_"+prevIter+"_"+i+".txt"));
				    StreamTokenizer stok = new StreamTokenizer(r);
				    stok.parseNumbers();
				    stok.nextToken();				    
				    while (stok.ttype != StreamTokenizer.TT_EOF) {
				        if (stok.ttype == StreamTokenizer.TT_NUMBER){
				           fitness[i]=stok.nval;
				        }			          
				        stok.nextToken();
				      }
				    System.out.println(i+"  ====================");
				    System.out.println(fitness[i]);
				    r.close();
				    
				    count =0;
					r = new BufferedReader(new FileReader(exppath+"params_"+prevIter+"_"+i+".txt"));
				    StreamTokenizer stok1 = new StreamTokenizer(r);
				    stok1.parseNumbers();
				    stok1.nextToken();				    
				    while (stok1.ttype != StreamTokenizer.TT_EOF) {
				        if (stok1.ttype == StreamTokenizer.TT_NUMBER){
				           prevPop[i][count]=stok1.nval;
				           count++;
				        }			          
				        stok1.nextToken();
				      }
				
				    }
				
				r = new BufferedReader(new FileReader(exppath+"sigma_"+prevIter+".txt"));
			    StreamTokenizer stok = new StreamTokenizer(r);
			    stok.parseNumbers();
			    stok.nextToken();
			    count=0;
			    while (stok.ttype != StreamTokenizer.TT_EOF) {
			        if (stok.ttype == StreamTokenizer.TT_NUMBER){
			           prevSigma[count]=stok.nval;
			           count++;
			        }			          
			        stok.nextToken();
			      }
			    r.close();
			    
				  
					
				}
				catch (Exception e)
				{
					e.printStackTrace();
					System.out.println("No such file exists.");
				}
				
				CMAEvolutionStrategy cma = new CMAEvolutionStrategy();
				cma.readProperties(); // read options`, see file CMAEvolutionStrategy.properties
				cma.setDimension(11); // overwrite some loaded properties
				
				//System.out.println(initSigma.length);
				cma.setInitialStandardDeviations(prevSigma); // also a mandatory setting
				cma.updateDistribution(prevPop, fitness);
				double[][] pop = cma.samplePopulation(); // get a new population of solutions
				//writeParams(pop,cma.s,curIter,11);	
				
			
			
		}
					
		
		/*
		EuclideanNew fitfun = new EuclideanNew();

		// new a CMA-ES and set some initial values
		CMAEvolutionStrategy cma = new CMAEvolutionStrategy();
		cma.readProperties(); // read options`, see file CMAEvolutionStrategy.properties
		cma.setDimension(11); // overwrite some loaded properties
		double initX[]={0.1,-0.05,-4,1.2,7,8,-99,-0.6,0.8,3.4,1.5};
		cma.setInitialStandardDeviation(0.2); // also a mandatory setting 
		double[] fitness = cma.init();  // new double[cma.parameters.getPopulationSize()];

		while(cma.stopConditions.getNumber() == 0) {

            // --- core iteration step ---
			System.out.println("Hello------>"+cma.sigma);
			double[][] pop = cma.samplePopulation(); // get a new population of solutions
			System.out.println(pop[0].length);
			for(int i = 0; i < pop.length; ++i) {    // for each candidate solution i
            	fitness[i] = fitfun.valueOf(pop[i]); // fitfun.valueOf() is to be minimized
			}
			cma.updateDistribution(fitness);         // pass fitness array to update search distribution
			
    		}*/
	} // main  
} // class
