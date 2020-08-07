package cma.examples;
class EuclideanNew  { // meaning implements methods valueOf and isFeasible
	public double valueOf (double[] x) {
		double best[]={1.5,1.2,1.7,2.4,2.1,3.0,1.6,1.2,2.65,2.13,3.8};
		double res = 0;
		
		for (int i = 0; i < x.length-1; ++i)
		{
			//System.out.print(x[i]+"   ");
			res += (best[i]-x[i])*(best[i]-x[i]);
		}
		System.out.println();
		res = Math.sqrt(res);
		return res;
	}
}

