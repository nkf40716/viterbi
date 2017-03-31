#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "FeaturePool.h"

const double kLogSmallProb = -30000000.0;

extern double ComputeGaussConst(float var[], int dim);
extern int InitModelTable(float model[]);
extern int GetDim();
extern int GetNumTotModel();
extern int GetNumTotState();
extern int GetModelSize();
extern int GetGaussID(int stateID, int mixtureIndex);
extern int GetModelStateList(int modelID, int stateIdList[]);
extern float *GetWordModel(int modelID);
extern float *GetState(int stateID);
extern double StateScore(int stateID, char x[], int dim);
extern int ReleaseModelTable();


static int MaxProb(float *A,float *B,float *C)
{
	if (C == NULL)
	{
		if (*A > *B)
			return 1;
		else if (*A<=kLogSmallProb && *B<=kLogSmallProb)
			return -1;
		else 
			return 2;
	}
	else if (*A > *B)
	{
		if (*A > *C)
			return 1;
		else
			return 3;
	} 
	else if (*A<=kLogSmallProb && *B<=kLogSmallProb && *C<=kLogSmallProb)
		return -1;
	else
	{
		if (*B > *C)
			return 2;
		else
			return 3;
	}
}


static int Min(int A,int B,int C)
{
	if (A<B)
	{
		if (A<C)
			return 1;
		else
			return 3;
	} 
	else
	{
		if (B<C)
			return 2;
		else
			return 3;
	}
}




int ReadModel(const char *pcModFilePath, float *pfModel)
{
	FILE *fp;
	fp = fopen(pcModFilePath,"rb");

	if(!fp){
		printf("\nCan't open model file!\n");
		exit(-1);
	}

	fseek(fp, 0, SEEK_END);
	int fpSize = (int)ftell(fp);
	rewind(fp);

	if (pfModel != NULL)
		fread(pfModel, 1, fpSize, fp);
	
	fclose(fp);	
	return fpSize;    //return file size (bytes)
}



int Viterbi_Detector(int nTotalState, int nTotalModel, int *pnStateIdListBuf, int *StartState, int *EndState, 
			  int nFrames, int nDim, char pcFP[], int *pResult)
{
	// allocate buffer
	int i,j;
	float *pScore = (float*)malloc(sizeof(float)*nTotalState);
	int *pBK = (int*)malloc(sizeof(int)*nTotalState);
	int *ModelIndexList = (int*)malloc(sizeof(int)*nFrames);
	int *DuList = (int*)malloc(sizeof(int)*nFrames);

	// Initialize first column
    for (j = 0; j < nTotalState; j++) 
	{
		pScore[j] = (float)kLogSmallProb;
		pBK[j] = 0;
    }

	for (j = 0; j <= nTotalModel; j++) 
		pScore[StartState[j]] = (float)StateScore(pnStateIdListBuf[StartState[j]], pcFP, nDim);
	
	//-----------Viterbi iteration-----------
	int t = 1;
	int m,nRow;
	float MaxEndScore = pScore[0];
	int MaxEndModelIndex = 0;

	ModelIndexList[0] = 0;
	DuList[0] = 0;

	int StateIdList[100];
	pcFP += nDim;
	for (t = 1; t < nFrames; t++, pcFP += nDim)
	{
		//for silence state
		pScore[0] = MaxEndScore + (float)StateScore(0, pcFP, nDim);
		pBK[0] = 0;
		nRow = nTotalState - 1;

		for (m = nTotalModel-1; m >= 0 ; m--)	//for all model
		{
			int nState = GetModelStateList(m, StateIdList);

			for(j = nState-1; j > 0; j--)
			{
				if ( (pBK[nRow] > 55) && (pBK[nRow-1] > 55) )	//Max duration
				{
					pScore[nRow] = (float)kLogSmallProb;
					pBK[nRow] = 0;
				}
				else if (MaxProb(&pScore[nRow], &pScore[nRow-1], NULL) == -1)
				{
					pScore[nRow] = (float)kLogSmallProb;
					pBK[nRow] = pBK[nRow] +1;
				}
				else if (MaxProb(&pScore[nRow], &pScore[nRow-1], NULL) == 1)
				{
					pScore[nRow] = pScore[nRow] + (float)StateScore(StateIdList[j], pcFP, nDim);
					pBK[nRow] = pBK[nRow] +1;
				}
				else
				{
					pScore[nRow] = pScore[nRow-1] + (float)StateScore(StateIdList[j], pcFP, nDim);
					pBK[nRow] = pBK[nRow-1] + 1;
				}
				nRow--;
			}
			pScore[nRow] = MaxEndScore + (float)StateScore(StateIdList[j], pcFP, nDim);
			pBK[nRow] = 0;
			nRow--;
		}	

		//Beam Pruning
		float MaxScore = (float)kLogSmallProb;
		for (j = 0; j < nTotalState; j++)
		{
			if (pScore[j] > MaxScore)
				MaxScore = pScore[j];	
		}

		float Beam = MaxScore - 100;
		for (j = 0; j < nTotalState; j++)
		{
			if (pScore[j] < Beam)
				pScore[j] = (float)kLogSmallProb;	
		}

		//find the max score for all model's ending state
		MaxEndScore = (float)kLogSmallProb;	
		for (j = 0; j <= nTotalModel; j++)
		{
			if (pScore[EndState[j]] > MaxEndScore && (pBK[EndState[j]] > 15 || j == 0))		//Min duration
			{
				MaxEndScore = pScore[EndState[j]];
				MaxEndModelIndex = j;		// MaxEndModelIndex - 1 == Word Model ID,      ( Index == 0 ) is silence
			} 
		}

		ModelIndexList[t] = MaxEndModelIndex;
		DuList[t] = pBK[EndState[MaxEndModelIndex]];

	}
	free(pScore);
	free(pBK);

	//Back Track
	t = nFrames - 1;
	int nDu = 0 , nCnt = 0;
	int *pResult_temp = (int*)malloc(sizeof(int)*100);
	while (t >= 0)
	{
		nDu = DuList[t];
		if (nDu != 0)
		{
			int temp = ModelIndexList[t];
			t -= nDu; 
			pResult_temp[nCnt] = temp - 1;		// temp - 1 == Word Model ID
			nCnt ++;
		}
		t --;
	}

	for (i = 0,j = nCnt-1; i < nCnt,j >= 0; i++,j--)
		pResult[i] = pResult_temp[j];


	free(ModelIndexList);
	free(DuList);
	free(pResult_temp);

	return nCnt;	// nCnt == size of result
}



int Edit_distance(int Ref_length, int *Ref, int Res_length, int *Result, int *pInsDelSub)
{
	int **D;
	int **pBK;
	int i,j;

	// allocate distance buffer
	D = new int * [Res_length+1];
	pBK = new int * [Res_length+1];
	for (i = 0; i <= Res_length; i++)
	{
		D[i] = new int [Ref_length+1];
		pBK[i] = new int [Ref_length+1];
	}

	// Initialize first column and first row
	for (i = 0; i <= Res_length; i++)
	{
		D[i][0] = i;
		pBK[i][0] = 1;
	}
	for (j = 0; j <= Ref_length; j++)
	{
		D[0][j] = j;
		pBK[0][j] = 1;
	}

	// Edit distance iteration
	int A,B,C;
	for (i = 1; i <= Res_length; i++)
	{
		for (j = 1; j <= Ref_length; j++)
		{
			if (Result[i-1] == Ref[j-1])
			{
				D[i][j] = D[i-1][j-1];
				pBK[i][j] = 0;
			}
			else
			{	
				A = D[i-1][j];
				B = D[i][j-1];
				C = D[i-1][j-1];
				if (Min(A,B,C) == 1)	//Insertion
				{
					D[i][j] = A + 1;
					pBK[i][j] = 1;

				} 
				else if (Min(A,B,C) == 2)	//Deletion 
				{
					D[i][j] = B + 1;
					pBK[i][j] = 2;

				} 
				else	//Substitution 
				{
					D[i][j] = C + 1;
					pBK[i][j] = 3;
				}
			}
		}
	}

	int nDis = D[Res_length][Ref_length];

	//Back Track
	i = Res_length;
	j = Ref_length;
	while (i>0 && j>0)
	{
		if (pBK[i][j] == 1)
		{
			*pInsDelSub += 1;	//ins
			i--;
		}
		else if (pBK[i][j] == 2)
		{
			*(pInsDelSub+1) += 1;	//del
			j--;
		}
		else if (pBK[i][j] == 3)
		{
			*(pInsDelSub+2) += 1;	//sub
			i--;
			j--;
		}
		else
		{
			i--;
			j--;
		}
	}

	for (i = 0; i <= Res_length; i++)
	{
		delete [] D[i];
		delete [] pBK[i];
	}
	delete [] D;
	delete [] pBK;


	return nDis;
}



int main(int argc, char* argv[])
{
	double START,END;
	START = clock();

	if(argc != 3){
		printf("Usage : %s <Model> <Testing FeaPool>\n", argv[0]);
		exit(-1);
	}

	//Read mod file
	char *pFilePath = argv[1];
	int fpSize = ReadModel(pFilePath,NULL);
	float *pfModel = (float *)malloc(fpSize);
	ReadModel(pFilePath,pfModel);
    InitModelTable(pfModel);
	free(pfModel);
	int nDim = GetDim();
	int nState = GetNumTotState();
	int nTotalModel = GetNumTotModel();


	//Word Model StateList to one dimension Array StateIdListBuf
	int *pStateIdList = (int*)malloc(sizeof(int)*nState);
 	int *StateIdListBuf = (int*)calloc(nState, sizeof(int));
	int *StartState = (int*)calloc(nTotalModel+1, sizeof(int));
	int *EndState = (int*)calloc(nTotalModel+1, sizeof(int));
	int i,j;
	int nWordModState;
	int nTotalState = 1;

	for (i=0;i<nTotalModel;i++)
	{
		nWordModState = GetModelStateList(i,pStateIdList);
		StartState[i+1] = *pStateIdList;
		EndState[i+1] = *(pStateIdList + nWordModState - 1);
			
		for(j=0;j<nWordModState;j++)
		{
			StateIdListBuf[nTotalState] = *(pStateIdList + j);
			nTotalState++;
		}	        
	}
 	free(pStateIdList);


	//Read fp file
	FeaturePool *fea;
	fea = new FeaturePool(argv[2]);
	int order = fea->GetDim();
	char *pcVectBuf = (char *)malloc(sizeof(char)*order*2000);
	char *pcRefBuf = (char *)malloc(sizeof(char)*100);
	int nUtterance = fea->GetNumUtterance();

	char *cRef;
	int *pResult = (int*)malloc(sizeof(int)*100);
	int *pIDS = (int*)calloc(3,sizeof(int));
	int totalDis = 0, num = 0, Sen = 0;

	for (i=0; i<nUtterance; i++)
	{
		int nVecframe = fea->GetVector(i,pcVectBuf);
 		fea->GetReference(i,pcRefBuf);

		//cut Ref State ID
		cRef = strtok(pcRefBuf,"-");
		int RefID[50];
		int Ref_length = 0;
		while (cRef != NULL)
		{
			if(*cRef != 'S')
			{
				RefID[Ref_length] = atoi(cRef);
				Ref_length++;
			}
			cRef = strtok(NULL,"-");
		}   
		
		int Res_length = Viterbi_Detector(nTotalState,nTotalModel,StateIdListBuf,StartState,EndState,
			nVecframe,nDim,pcVectBuf,pResult);

		int Dis = Edit_distance(Ref_length,RefID,Res_length,pResult,pIDS);

		if (Dis != 0)
		{
			Sen++;
			printf("i=%d, Ref=",i+1);
			for (j=0;j<Ref_length;j++)	
				printf("%d ",RefID[j]);
			printf(", Res=");
			for (j=0;j<Res_length;j++)	
				printf("%d ",pResult[j]);
			printf(" <X>\n");
		}

		totalDis += Dis;
		num += Ref_length;
		
	}

	float CharErr = ((float)totalDis / (float)num) * 100;
	float SenErr = ((float)Sen / (float)nUtterance) * 100;


	printf("Utterance = %d , SenErr = %d\n",nUtterance,Sen);
	printf("Ins=%d , Del=%d , Sub=%d , CharErr=%.2f%% , SenErr=%.2f%%\n",pIDS[0],pIDS[1],pIDS[2],CharErr,SenErr);
	

	ReleaseModelTable();
	fea->~FeaturePool();
	free(pResult);
	free(pcRefBuf);
	free(pIDS);
	free(StateIdListBuf);
	free(StartState);
	free(EndState);
	free(pcVectBuf);
	
	END = clock();
	printf("\nTime = %.2f\n",(END - START) / CLOCKS_PER_SEC);

	return 0;
}





