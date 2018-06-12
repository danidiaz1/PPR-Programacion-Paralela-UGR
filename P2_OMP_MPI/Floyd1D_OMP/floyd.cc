#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"
#include <omp.h>
#include <cstdlib>
#include <cmath>

using namespace std;

//**************************************************************************

int main (int argc, char *argv[])
{

if (argc != 3) {
   cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" <<
   " <nº de hebras a lanzar" << endl;
   return(-1);
}

Graph G;
G.lee(argv[1]);         // Read the Graph
//cout << "EL Grafo de entrada es:"<<endl;
//G.imprime();

int P = atoi(argv[2]); // Nº de hebras a lanzar
omp_set_num_threads(P);

int nverts=G.vertices;
int i, k, j, vikj, fila_k_esima[nverts];

double t1=omp_get_wtime();
#pragma omp parallel private(fila_k_esima, k, j, vikj)
{

   for (k=0; k < nverts; k++)
   {
   		for (j = 0; j < nverts; j++)
        	fila_k_esima[j] = G.arista(k,j);

        #pragma omp for schedule(static)
          for (i = 0; i < nverts; i++)
              for (j = 0; j < nverts; j++)
                if (i!=j && i!=k && j!=k){
                   vikj=min(G.arista(i,k)+fila_k_esima[j], G.arista(i,j));
                   G.inserta_arista(i,j,vikj);
                }
   }
}

double  t2=omp_get_wtime();
t2 = t2-t1;

cout << "Tiempo OMP-1D: " << t2 << endl << endl;

//cout << endl<<"EL Grafo con las distancias de los caminos más cortos es:"<<endl;
//G.imprime();

return 0;
}
