import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Scanner;


public class ID3 {

	public static void main(String[] args) throws IOException {
		int numOfFeatures = 3;
		int numOfTarget = 2;
		Scanner sc = new Scanner(System.in);
		System.out.println("Enter names of the files dataset input-partition output-partition");
		String datafile = sc.next();
		String inputPartition = sc.next();
		String outputPartition = sc.next();
		sc.close();
		sc = new Scanner(new FileReader(datafile));
		int m = sc.nextInt();
		int n = sc.nextInt();
		int data[][] = new int[m][n];
		for(int i = 0 ; i < m ; i++){
			for ( int j = 0 ; j < n ; j++){
				data[i][j] = sc.nextInt();
			}
		}
		sc.close();
		BufferedReader br = new BufferedReader(new FileReader(inputPartition));
		String line ;
		LinkedHashMap<String,ArrayList<Integer>> map = new LinkedHashMap<String,ArrayList<Integer>>();
		while ( ( line = br.readLine())!= null){
			String[] lineArray = line.split(" ");
			String key = lineArray[0];
			ArrayList<Integer> list = new ArrayList<Integer>();
			for(int i=1;i<lineArray.length; i++)
			{
				list.add(Integer.parseInt(lineArray[i]));
			}
			map.put(key, list);
		}
		System.out.println("Partition "+ map + " was replaced with partitions");
		double totalEntropy = 0.0;
		//For each Partition
		double[] partitionValue = new double[map.size()];
		int[] GainIndex=new int[map.size()];
		int maxGainIndex = -1;
		int count = 0;
		for(String key : map.keySet())
		{
			double maxGain = -999999999.0;
			int[] target = new int[numOfTarget];
			ArrayList<Integer> partition = map.get(key);
			for (int instance : partition) {
				target[data[instance-1][n-1]]++;
			}
			double e1 = target[0]*1.0/(target[0]+target[1])*1.0;
			double e2 = target[1]*1.0/(target[0]+target[1])*1.0;
			totalEntropy = entropy(e1,e2 );
			//For Each Column or Feature
			double[] entropyCol = new double[3];
			double[] gainCol = new double[3];
			for(int  col = 0 ; col < 3; col++)
			{
				int[] nF = new int[numOfFeatures];//It contains 0,1,2
				for (int instance : partition) {
					nF[data[instance-1][col]]++;
				}
				double[] entropyF = new double[numOfFeatures];//Entropies of each Feature 0,1,2
				//For each column 
				for(int f = 0; f < 3 ; f++){
					int[] nTarget = new int[2];
					for (int instance : partition) {
						if( data[instance-1][col] == f){
							nTarget[data[instance-1][n-1]]++;
						}
					}
					if( nF[f] == 0) {
						entropyF[f] = 0;
					}else{
						entropyF[f] = entropy(nTarget[0]*1.0/nF[f]*1.0,nTarget[1]*1.0/nF[f]*1.0);
					}
				}
				entropyCol[col] = (nF[0]*entropyF[0] + nF[1]* entropyF[1] + nF[2]*entropyF[2])/(nF[0]+nF[1]+nF[2]);
				gainCol[col] = totalEntropy - entropyCol[col];
				if(gainCol[col] > maxGain){
					maxGain = gainCol[col];
					maxGainIndex = col;
				}
			}
			partitionValue[count] =  (partition.size()*1.0/m*1.0)*maxGain;
			GainIndex[count]=maxGainIndex;
			count++;
		}
		double maxPartitionValue = -99999999.0;
		int maxPartitionIndex = -1;
		for (int i = 0; i < partitionValue.length; i++) {
			if( partitionValue[i] > maxPartitionValue) {
				maxPartitionValue = partitionValue[i];
				maxPartitionIndex = i;
				maxGainIndex=GainIndex[i];
			}
		}
		count = 0;
		HashMap<String,ArrayList<Integer>> resultMap = new HashMap<String,ArrayList<Integer>>();
		String removeKey = "";
		for( String key : map.keySet())
		{
			if(count == maxPartitionIndex){
				ArrayList<Integer> resultList = map.get(key);
				for( int instance : resultList){
					if( resultMap.get(key+data[instance-1][maxGainIndex]) != null){
						ArrayList<Integer> list = resultMap.get(key+data[instance-1][maxGainIndex]);
						list.add(instance);
						
					}
					else{
						ArrayList<Integer> list = new ArrayList<Integer>();
						list.add(instance);
						resultMap.put(key+data[instance-1][maxGainIndex]+"", list);
					}
				}
				removeKey = key;
			}
			count++;
		}
		map.remove(removeKey);
		map.putAll(resultMap);
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputPartition));
		for( String key : map.keySet()){
			bw.write(key  + " " + Arrays.toString(map.get(key).toArray())+"\n");
			System.out.print(key  + " " + Arrays.toString(map.get(key).toArray()));
		}
		System.out.println(" using feature" + (maxGainIndex+1) );
		bw.close();
		br.close();
	}

	private static double entropy(double e1, double e2) {
		if ( e1 == 0 || e2 == 0 ) return 0.0;
		else {
			return e1*(Math.log(1/e1)/Math.log(2))+ e2*(Math.log(1/e2)/Math.log(2));
		}
	}
}
