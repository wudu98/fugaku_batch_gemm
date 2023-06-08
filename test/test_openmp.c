#include <omp.h>
#include <stdio.h>

void report_num_threads(int level){
	#pragma omp single
	{
		printf("Level: %d, number of threads = %d\n", level, omp_get_num_threads());
	}
}

int main() {
    int num_threads = 48;
    int num_regions = 4;
    int threads_per_region = 12;

    omp_set_nested(1);
    omp_set_num_threads(num_threads);

    #pragma omp parallel num_threads(num_regions)
    {
        int region_id = omp_get_thread_num();

        report_num_threads(1);
        omp_set_num_threads(threads_per_region);

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            report_num_threads(2);
            printf("Region %d, Thread %d\n", region_id, thread_id);
        }

        omp_set_num_threads(num_threads);
    }

    return 0;
}
