/* Replay exported-model confidences through the real isolated bell event path. */
#include "../app/detector.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    int expected = atoi(argv[2]);
    for (int enabled = 0; enabled <= 1; enabled++) {
        FILE* f = fopen(argv[1], "r");
        if (!f) return 2;
        detector_t d; det_config_t c;
        detector_init(&d); detector_config_defaults(&c);
        c.bell_triggers = enabled != 0; c.horn_triggers = false;
        c.sensitivity = 10; c.min_level_db = -90;
        double now = 100; float head, bell, level;
        while (fscanf(f, "%f,%f,%f", &head, &bell, &level) == 3) {
            now += .48;
            /* Head and horns intentionally zero: success must be bell-only. */
            detector_update_sources(&d, &c, now, now, .48, 0, 0, 0, level, 0, bell);
        }
        fclose(f);
        if ((d.events_total != 0) != (enabled && expected)) {
            fprintf(stderr, "%s enabled=%d unexpected events=%llu\n", argv[1], enabled, (unsigned long long)d.events_total);
            return 1;
        }
        if (d.events_total && (!d.cur.bell || d.cur.horn || d.cur.crossing)) return 1;
    }
    printf("PASS %s (bell-only and toggle-off)\n", argv[1]);
    return 0;
}
