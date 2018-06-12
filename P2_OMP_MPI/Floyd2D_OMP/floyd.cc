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
omp_set_dynamic(0);
omp_set_num_threads(P);

int nverts=G.vertices;

const int fils_cols = nverts/sqrt(P), // Filas, columnas de la submatriz de cada proceso
    bsize = fils_cols*fils_cols, // Tamaño de bloque
    grid_dim= sqrt(P); // Número de procesos que hay en cada fila/columna del
                      // grid conceptual de procesos

int sub_matriz[fils_cols][fils_cols],
      fila_k_esima[nverts],
      columna_k_esima[nverts],
      fila_k_esima_privada[fils_cols],
      columna_k_esima_privada[fils_cols];


double t1=omp_get_wtime();
#pragma omp parallel private(sub_matriz, fila_k_esima_privada, columna_k_esima_privada)
{
   int t_id = omp_get_thread_num();
   int i, j, vikj;

   // indices globales iniciales de fila y columna de cada hebra
   int i_global = (t_id/grid_dim)*fils_cols;
   int j_global;

   // Copia privada por parte de cada hebra de su parte de la matriz
   for (i = 0; i < fils_cols; i++, i_global++){
       j_global = (t_id%grid_dim)*fils_cols;
       for (j = 0; j < fils_cols; j++, j_global++)
          sub_matriz[i][j] = G.arista(i_global,j_global);
   }

   // Floyd
   for (int k=0; k < nverts; k++)
   {
      i_global = (t_id/grid_dim)*fils_cols;
      j_global = (t_id%grid_dim)*fils_cols;

      // Comprobamos si la fila y columna k-ésima están en nuestra submatriz
      bool tienes_fil_k_esima = k >= i_global && k < i_global+fils_cols;
      bool tienes_col_k_esima = k >= j_global && k < j_global+fils_cols;

      // Copia en memoria compartida de la fila k-ésima
      if (tienes_fil_k_esima)
        memcpy(&fila_k_esima[j_global], &sub_matriz[k%fils_cols][0], sizeof(int)*fils_cols);

      // Copia en memoria compartida de la columna k-ésima
      if (tienes_col_k_esima)
        for (i = 0; i < fils_cols; i++, i_global++)
            columna_k_esima[i_global] = sub_matriz[i][k%fils_cols];

      // Esperamos a que se copie en memoria compartida
      #pragma omp barrier

      // Privatizamos las partes necesarias de la fila y columna k_esima en cada hebra
      i_global = (t_id/grid_dim)*fils_cols;
      j_global = (t_id%grid_dim)*fils_cols;

      // fila k-ésima privada
      for (i = 0; i < fils_cols; i++, j_global++)
        fila_k_esima_privada[i] = fila_k_esima[j_global];

      for (i = 0; i < fils_cols; i++, i_global++)
        columna_k_esima_privada[i] = columna_k_esima[i_global];

      // algoritmo de Floyd
      i_global = (t_id/grid_dim)*fils_cols;

      for (i = 0; i < fils_cols; i++, i_global++){
        j_global = (t_id%grid_dim)*fils_cols;
        for (j = 0; j < fils_cols; j++, j_global++)
          if (i_global!=j_global && i_global!=k && j_global!=k){
             vikj=min(fila_k_esima_privada[j]+columna_k_esima_privada[i], sub_matriz[i][j]);
             sub_matriz[i][j]=vikj;
          }
      }

      // Para que no se empiece el siguiente "k" hasta que no terminen todas las hebras,
      // ya que alguna podría alterar el estado de las variables compartidas de
      // fila y columna k-ésima antes de que cada hebra las privatice en el "k" anterior
      #pragma omp barrier
   }

   // Escritura final del grafo por parte de cada hebra
   i_global = (t_id/grid_dim)*fils_cols;

   for (i = 0; i < fils_cols; i++, i_global++){
      j_global = (t_id%grid_dim)*fils_cols;
      for (j = 0; j < fils_cols; j++, j_global++)
         G.inserta_arista(i_global,j_global,sub_matriz[i][j]);
   }

   #pragma omp barrier
}

double  t2=omp_get_wtime();
t2 = t2-t1;

cout << "Tiempo OMP-2D: " << t2 << endl << endl;

//cout << endl<<"EL Grafo con las distancias de los caminos más cortos es:\n";
//G.imprime();

return 0;
}
