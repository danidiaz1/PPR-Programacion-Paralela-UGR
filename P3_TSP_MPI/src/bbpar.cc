/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "../lib/libbb.h"

using namespace std;

unsigned int NCIUDADES;
int rank, size, siguiente, anterior;
bool token_presente;

MPI_Comm comunicadorCarga;	// Para la distribución de la carga
MPI_Comm comunicadorCota;	// Para la difusión de una nueva cota superior detectada

main (int argc, char **argv) {
  /***********************************************************/
  /*********** Inicialización del entorno MPI ****************/
  /***********************************************************/
  int color_carga = 0;
  int color_cota = 1;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Inicializamos comunicadores
  MPI_Comm_split(MPI_COMM_WORLD, color_carga, rank, &comunicadorCarga);
  MPI_Comm_split(MPI_COMM_WORLD, color_carga, rank, &comunicadorCota);

  siguiente = (rank+1)%size;
  anterior = (rank-1+size)%size;

	switch (argc) {
		case 3:		NCIUDADES = atoi(argv[1]);
					break;
		default:	cerr << "La sintaxis es: bbpar <tamanio> <archivo>" << endl;
					exit(1);
					break;
	}

  // Variables de cada proceso
	int** tsp0 = reservarMatrizCuadrada(NCIUDADES);
	tNodo	nodo,         // nodo a explorar
			lnodo,        // hijo izquierdo
			rnodo,        // hijo derecho
			solucion;     // mejor solucion
	bool fin,        // condicion de fin
		nueva_U;       // hay nuevo valor de c.s.
	int  U;             // valor de c.s.
	int iteraciones = 0;
	tPila pila;         // pila de nodos a explorar

	U = INFINITO;                  // inicializa cota superior
	InicNodo (&nodo);              // inicializa estructura nodo


  if (rank == 0){
    LeerMatriz (argv[2], tsp0);   // lee matriz de fichero
    token_presente = true;
  }

  // Compartimos la matriz con el resto
  MPI_Bcast (&tsp0[0][0], NCIUDADES*NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);

  // Medida de tiempo
  MPI_Barrier(MPI_COMM_WORLD);
  double t=MPI_Wtime();

  // Los procesos que no son el 0 no tienen nodos, tienen que llamar
  // al equilibrado de carga
  if (rank != 0) {
    token_presente = false;
    Equilibrar_Carga(pila, fin, solucion);
    if (!fin) pila.pop(nodo);
  }



  /***********************************************************/
  /**************** Ciclo del Branch&Bound *******************/
  /***********************************************************/

  fin = Inconsistente(tsp0);

	while (!fin) {
		Ramifica (&nodo, &lnodo, &rnodo, tsp0);
		nueva_U = false;
		if (Solucion(&rnodo)) {
			if (rnodo.ci() < U) {    // se ha encontrado una solucion mejor
				U = rnodo.ci();
				nueva_U = true;
				CopiaNodo (&rnodo, &solucion);
			}
		}
		else {                    //  no es un nodo solucion
			if (rnodo.ci() < U) {     //  cota inferior menor que cota superior
				if (!pila.push(rnodo)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}
		if (Solucion(&lnodo)) {
			if (lnodo.ci() < U) {    // se ha encontrado una solucion mejor
				U = lnodo.ci();
				nueva_U = true;
				CopiaNodo (&lnodo,&solucion);
			}
		}
		else {                     // no es nodo solucion
			if (lnodo.ci() < U) {      // cota inferior menor que cota superior
				if (!pila.push(lnodo)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}

    //Difusion_Cota_Superior(U, nueva_U);
		if (nueva_U) pila.acotar(U);

    Equilibrar_Carga(pila, fin, solucion);
		if (!fin) pila.pop(nodo);

		iteraciones++;
	}
  MPI_Barrier(MPI_COMM_WORLD);
  t=MPI_Wtime()-t;

  printf("Proceso %d. Numero de iteraciones = %d\n", rank, iteraciones);
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0){
  	printf ("\nSolucion: \n");
  	EscribeNodo(&solucion);
    cout<< "Tiempo gastado= "<<t<<endl;
  }

	liberarMatriz(tsp0);

  MPI::Finalize();

  exit(0);
}
