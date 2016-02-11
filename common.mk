# Common definition shared by all workloads


OUTPUT_LOG=output.log
EXPECTED_LOG=ref_output
DIFF_LOG=diff.log


help:
	@echo " USAGE: make [OPTIONS] <TARGET>"
	@echo ""
	@echo " Valid <TARGET> are:"
	@echo "            all: build all binaries"
	@echo "            run: run application"
	@echo "         verify: verify application"
	@echo "          clean: clean up generated files"
	@echo ""
	@echo " Check documents for supported [OPTIONS]"
	
reset_generated_dir:
	@if [ -n "${GENERATED_DIRS}" ]; then \
          rm -rf ${GENERATED_DIRS}; \
          mkdir ${GENERATED_DIRS};  \
        fi

run: ${TARGET} reset_generated_dir
	@if [ -n "${TARGET}" ]; then \
          echo "Running ${TARGET}, output in ${OUTPUT_LOG}"; \
          ./${TARGET} ${RUN_ARGS} ${PERF_ARGS} > ${OUTPUT_LOG} 2>&1; \
	fi

verify: 
	@echo "VERIFY-run [${TARGET}]"
	@-make clean;
	@echo "ReCompile with options: OUTPUT=1 VERIFY=1";
	@-make OUTPUT=1 VERIFY=1 all;
	@echo "Running...";
	@./${TARGET} ${RUN_ARGS}  > ${OUTPUT_LOG} 2>&1;
	@${ROOT}/scripts/compare.sh ${OUTPUT_LOG} ${EXPECTED_LOG} ${DIFF_LOG}

clean:
	@-/bin/rm -rf ${ALL_TARGETS} ${GENERATED_DIRS} *.o *~ core core.* ${OUTPUT_LOG} ${DIFF_LOG}

