#include <pthread.h>
#include <stdio.h>

void *codigo_del_hilo(void *id) {
  int i;
  for (i = 0; i < 50; i++)
    printf("\n Soy el hilo: %d, iter = %d", *(int*)id, i);
  pthread_exit(id);
}

int main(void) {
  pthread_t hilo;
  int id = 245;
  int *salida;

  pthread_create(&hilo, NULL, codigo_del_hilo, &id);
  pthread_join(hilo, NULL);
  printf("\n Hilo %d terminado \n", *salida);

  return (0);
}