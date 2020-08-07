package cma.examples;
import java.io.*;

//Command Line Iterations : IterNo
public class EvaluateParams {
	static Reader r;
	static PrintWriter pri;
	static String exppath = "/home/suyog/cmaes/temp/";
	static double pop[][] = new double[11][11];
	static double fitness[] = new double[11];
	public static void main(String[] args) {
		
		int curIter = Integer.parseInt(args[0]);
		System.out.println(curIter);
		EuclideanNew ec = new EuclideanNew();
		
		try
		{
			
			int count=0;
			for(int i =0;i<11;i++)
				{
				//System.out.println(exppath+"params_"+curIter+"_"+i+".txt");
				r = new BufferedReader(new FileReader(exppath+"params_"+curIter+"_"+i+".txt"));
			    StreamTokenizer stok = new StreamTokenizer(r);
			    stok.parseNumbers();
			    stok.nextToken();
			    count=0;
			    while (stok.ttype != StreamTokenizer.TT_EOF) {
			        if (stok.ttype == StreamTokenizer.TT_NUMBER){
			      //     System.out.println(stok.nval);
			           pop[i][count]=stok.nval;
			           //System.out.println(pop[i][count]);
			           count++;
			        }			          
			        stok.nextToken();
			      }
			    System.out.println(pop[i].length);
			    fitness[i]= ec.valueOf(pop[i]);
			    System.out.println(i+"  ====================");
			    System.out.println(fitness[i]);
			    System.out.println("====================");
			    pri = new PrintWriter(new FileWriter(exppath+"value_"+curIter+"_"+i+".txt", true));
				pri.println(fitness[i]);
				pri.close();
			    r.close();
				}
				
			}
			catch (Exception e)
			{
				e.printStackTrace();
				System.out.println("No such file exists.");
			}						
		
	} // main  
} // class
