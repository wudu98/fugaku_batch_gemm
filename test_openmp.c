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

    printf("%d\n", omp_get_max_active_levels());
    omp_set_max_active_levels(2);
    printf("%d\n", omp_get_max_active_levels());
    omp_set_num_threads(num_threads);

    #pragma omp parallel num_threads(num_regions)
    {
        int region_id = omp_get_thread_num();

        omp_set_num_threads(threads_per_region);
        report_num_threads(1);

        #pragma omp parallel num_threads(num_regions)
        {
            report_num_threads(2);
        }

        omp_set_num_threads(num_threads);
    }

    return 0;
}
