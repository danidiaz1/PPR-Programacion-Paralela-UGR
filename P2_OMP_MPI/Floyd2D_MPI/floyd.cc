#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"
#include "mpi.h"
#include <cstdlib>
#include <cmath>

using namespace std;

//**************************************************************************

int main (int argc, char *argv[])
{

int rank, P;
MPI_Status estado;
MPI_Datatype MPI_BLOQUE, MPI_COLUMNA;
MPI_Comm COMM_HOR, COMM_VERT;

MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
MPI_Comm_size(MPI_COMM_WORLD, &P); // Obtenemos el numero total de hebras
MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador


if (argc != 2) {
   cerr << "Sintaxis: mpirun -np <nº hebras> " << argv[0] << " <archivo de grafo>"
   << endl;
   return(-1);
}


/****************************************/
/*          Lectura del grafo           */
/****************************************/
int nverts;
Graph G;

if (rank == 0){
  G.lee(argv[1]);         // Read the Graph
  //cout << "EL Grafo de entrada es:"<<endl;
  //G.imprime();

  nverts=G.vertices;
}

// Hacemos un broadcast del tamaño del grafo al resto de procesos
MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);

const int fils_cols = nverts/sqrt(P), // Filas, columnas de la submatriz de cada proceso
    bsize = fils_cols*fils_cols, // Tamaño de bloque
    grid_dim= sqrt(P); // Número de procesos que hay en cada fila/columna del
                      // grid conceptual de procesos

// Comunicadores horizontal y vertical
MPI_Comm_split(MPI_COMM_WORLD, rank/grid_dim, rank, &COMM_VERT);
MPI_Comm_split(MPI_COMM_WORLD, rank%grid_dim, rank, &COMM_HOR);

int i, j, vikj, posicion, fila_P, columna_P, comienzo,
    sub_matriz[fils_cols][fils_cols],
    *matriz_grafo = new int[nverts*nverts],
    fila_k_esima[fils_cols],
    columna_k_esima[fils_cols];



/*****************************************/
/*            Division y reparto         */
/*              de bloques               */
/*****************************************/
if (rank == 0){

  int * matriz_I = G.getMatriz(); // Matriz original que representa el grafo
  // Defino el tipo bloque cuadrado
  MPI_Type_vector (
    fils_cols,  // Tamaño del bloque
    fils_cols,  // distancia que separa cada bloque de otro
    nverts,     // numero de bloques
    MPI_INT,    // tipo de dato del vector
    &MPI_BLOQUE // variable que será el nuevo tipo
  );

  // Creo el nuevo tipo
  MPI_Type_commit (&MPI_BLOQUE);

  //Empaqueta bloque a bloque en el buffer de envío
  for (i=0, posicion=0; i<P; i++) {
    // Calculo la posicion de comienzo de cada submatriz
    fila_P = i/grid_dim;
    columna_P = i%grid_dim;
    comienzo = columna_P * fils_cols + fila_P * bsize * grid_dim;

    MPI_Pack (
      matriz_I + comienzo,          // Posicion inicial del los datos a enviar
      1,                            // Numero de elementos a enviar
      MPI_BLOQUE,                   // Tipo de dato a enviar
      matriz_grafo,                 // Buffer de recepción
      sizeof(int)*nverts*nverts,    // Tamaño del buffer de recepción en bytes
      &posicion,                    // Posicion del buffer de recepción
      MPI_COMM_WORLD                // Comunicador
    );
  }

}

// Distribuimos la matriz entre los procesos
MPI_Scatter (
  matriz_grafo,         // Inicio de los datos a repartir
  sizeof(int)*bsize,    // Cuantos datos enviamos a cada proceso en bytes
  MPI_PACKED,           // Tipo de dato a enviar
  sub_matriz,           // Dónde se recibirán los datos
  bsize,                // Número de datos a recibir
  MPI_INT,              // Tipo de dato a recibir
  0,                    // Proceso que hace el scatter
  MPI_COMM_WORLD        // Comunicador
);

// Sincronizamos los procesos para medir tiempos
MPI_Barrier(MPI_COMM_WORLD);

double t1=MPI_Wtime();




/****************************************/
/*                 Floyd                */
/****************************************/
int proc_filcol_k_esima;

// indices globales iniciales de fila y columna de cada proceso
int i_global;
int j_global;

for (int k=0; k < nverts; k++)
{
  i_global = (rank/grid_dim)*fils_cols;
  j_global = (rank%grid_dim)*fils_cols;

  // proceso que tiene la fila y columna_k_esima (dentro de sus comunicadores)
  proc_filcol_k_esima = k/fils_cols;

  // Comprobamos si la fila y columna k-ésima están en nuestra submatriz
  bool tienes_fil_k_esima = k >= i_global && k < i_global+fils_cols;
  bool tienes_col_k_esima = k >= j_global && k < j_global+fils_cols;

  if (tienes_fil_k_esima)
    memcpy(fila_k_esima, &sub_matriz[k%fils_cols][0], sizeof(int)*fils_cols);

  if (tienes_col_k_esima)
    for (i = 0; i < fils_cols; i++)
      columna_k_esima[i] = sub_matriz[i][k%fils_cols];

  // Broadcast de la fila y columna k-ésima
  MPI_Bcast(fila_k_esima, fils_cols, MPI_INT, proc_filcol_k_esima, COMM_HOR);
  MPI_Bcast(columna_k_esima, fils_cols, MPI_INT, proc_filcol_k_esima, COMM_VERT);

  i_global = (rank/grid_dim)*fils_cols;

  for (i = 0; i < fils_cols; i++, i_global++){
    j_global = (rank%grid_dim)*fils_cols;
    for (j = 0; j < fils_cols; j++, j_global++)
      if (i_global!=j_global && i_global!=k && j_global!=k){
         vikj=min(fila_k_esima[j]+columna_k_esima[i], sub_matriz[i][j]);
         sub_matriz[i][j]=vikj;
      }
  }
}

MPI_Barrier(MPI_COMM_WORLD);

double  t2=MPI_Wtime();
t2 = t2-t1;




/*****************************************************/
/*              Recogida de los datos                */
/*****************************************************/

MPI_Gather(
  sub_matriz,                   // Inicio de los datos a enviar
  bsize,                        // Número de datos a enviar desde cada proceso
  MPI_INT,                      // Tipo de dato a enviar
  matriz_grafo,                 // Dónde se recibirán los datos
  sizeof(int)*bsize,            // Cantidad de datos a recibir de cada proceso en bytes
  MPI_PACKED,                   // Tipo de dato a recibir
  0,                            // Proceso donde se reuniran los datos
  MPI_COMM_WORLD                // Comunicador
);

int * matriz_final = new int[nverts*nverts];

if (rank == 0){

  for (i=0, posicion=0; i<P; i++) {
    // Calculo la posicion de comienzo de cada submatriz
    fila_P = i/grid_dim;
    columna_P = i%grid_dim;
    comienzo = columna_P * fils_cols + fila_P * bsize * grid_dim;

    MPI_Unpack(
      matriz_grafo,                     // Inicio de los datos a desempaquetar
      sizeof(int)*nverts*nverts,        // Cantidad de datos en bytes
      &posicion,                        // Posicion del buffer de entrada
      matriz_final + comienzo,          // Posicion de inicio donde desempaquetar
      1,                                // Número de elementos a desempaquetar
      MPI_BLOQUE,                       // Tipo de elementos a desempaquetar
      MPI_COMM_WORLD                    // Comunicador
    );
  }

  // Libero el tipo bloque
  MPI_Type_free (&MPI_BLOQUE);

  cout << "Tiempo MPI-2D: " << t2 << endl << endl;

  /*cout << endl<<"EL Grafo con las distancias de los caminos más cortos es:\n";

  for(i=0;i<nverts;i++){
    cout << "A["<<i << ",*]= ";

    for(j=0;j<nverts;j++){
      if (matriz_final[i*nverts+j]==INF) cout << "INF";
      else cout << matriz_final[i*nverts+j];

      if (j<nverts-1) cout << ",";
      else cout << endl;
    }
  }*/

  delete matriz_final;
}

MPI_Barrier(MPI_COMM_WORLD);
delete matriz_grafo;
MPI_Finalize();

return 0;
}
