import subprocess
import numpy as np
import scipy.stats as st
import math
import time
import importlib.util
import pathlib
import sys
import os
import random
import datetime
import pandas
import signal
import shlex
import json
from typing import Callable

class data_container:
    def __init__(self, app_id, conv_goal, label, unit):
        self.app_id = app_id
        self.conv_run = 0
        self.label = label
        self.unit = unit
        self.conv_goal = conv_goal
        self.converged = False
        self.num_samples = []
        self.data = []

    def get_title(self):
        return str(self.app_id)+'_'+self.label+'_'+self.unit

    def md_to_list(self):
        return [self.app_id, self.label, self.unit, self.conv_goal, self.converged, self.conv_run]+self.num_samples


def check_CI(container_list, alpha, beta, converge_all, run):
    # confidence interval parameters: length of CI (two-tailed with 1-alpha confidence) is within beta of reported mean
    for container in container_list:
        # not converged yet and has to converge
        if (not container.converged) and (converge_all or container.conv_goal):
            n = len(container.data)
            if n <= 1:
                continue  # not converged
            mean = np.mean(container.data)
            sem = st.sem(container.data)
            if sem == 0:  # no variance in data
                container.converged = True  # converged
                container.conv_run = run
                continue
            CI_lb, CI_ub = st.t.interval(1-alpha, n-1, loc=mean, scale=sem)
            CI_length = CI_ub-CI_lb
            if CI_length < beta*mean:
                container.converged = True  # converged
                container.conv_run = run

    # check if all convergence flags
    check = True
    for container in container_list:
        if (converge_all or container.conv_goal):
            check = (check and container.converged)
    return check

def run_job(job, wlmanager, ppn):
    if job.num_nodes == 0:
        #TODO: Usa il logger!
        print(f"[ERROR] L'applicazione {job.id_num} ha 0 nodi allocati...")
        raise Exception(f"Application {job.id_num} has 0 allocated nodes.")
    cmd_string = wlmanager.run_job(job.node_list, ppn, job.run_app())
    if not cmd_string or cmd_string == "":
        cmd_string = "echo a > /dev/null"
    cmd = shlex.split(cmd_string)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    job.set_process(process)

def wait_timed(job, time):
    try:
        out, err = job.process.communicate(timeout=time)
        # if didn't time out
        job.set_output(out, err)
        return False
    except subprocess.TimeoutExpired:
        # if timed out
        end_job(job)
        return True


def end_job(job):
    # signal SIGUSR1 if ouput of endless is needed
    # TODO: Restora
    #if job.collect_flag:
    #    job.process.terminate()
    #else:
    #    job.process.kill()
    job.process.kill()
    out, err = job.process.communicate()
    job.set_output(out, err)


def c_allocation(apps, nodes_frame):
    for app in apps:
        nodes_list = list(filter(lambda ele: ele != '',
                          nodes_frame[:][app.id_num].tolist()))
        app.set_nodes(nodes_list)


def l_allocation(apps, node_list, split_absolute):  # linear allocation
    i = 0
    for app, split in zip(apps, split_absolute):
        app.set_nodes(node_list[i:i+split])
        i += split


def i_allocation(apps, node_list, split_absolute):  # interleaved allocation
    num_apps = len(apps)
    ending_condition = [0]*num_apps
    alloc_lists = [[] for _ in range(num_apps)]
    s_index = 0
    n_index = 0
    while split_absolute != ending_condition:
        if split_absolute[s_index] != 0:
            # TODO Far evitare che crashi quando n_index >= len(node_list)
            alloc_lists[s_index] += [node_list[n_index]]
            split_absolute[s_index] -= 1
            n_index += 1
        s_index = (s_index+1) % num_apps
    for app, alloc_list in zip(apps, alloc_lists):
        app.set_nodes(alloc_list)


def get_abs_split(allocation_split, num_apps, num_nodes):
    if allocation_split == 'e':  # equal split among all apps
        split_list = [100/num_apps]*num_apps
    else:
        split_list = [float(x) for x in allocation_split.split(':')]

    if sum(split_list) > 100:
        raise Exception("Splits percentages mustn't add up to more than 100.")
    '''
    if len(split_list) != num_apps:
        raise Exception('Number of applications ('+str(num_apps)+') is not equal to number of splits ('
                        + str(len(split_list))+')')
    '''
    if len(split_list) < num_apps:
        raise Exception('Number of applications ('+str(num_apps)+') is larger than number of splits ('
                    + str(len(split_list))+')')
    
    split_list = split_list[:num_apps]
    print("Split list: " + str(split_list))
    split_absolute = []
    for split in split_list[:-1]:
        split_a = int(math.ceil(num_nodes*split/100))
        split_absolute += [split_a]
        print("Split_a: " + str(split_a))
    if num_apps == 1:
        split_absolute = [int(math.ceil(num_nodes*split_list[0]/100))]
    else:
        # allocate all remaining nodes to the last application
        split_absolute += [num_nodes-sum(split_absolute)]
    print("Splits: " + str(split_absolute))
    return split_absolute


def log_meta_data(out_format, path, data_container_list, num_runs):
    num_samples_index = list(
        map(lambda x: 'Run '+str(x), list(range(1, num_runs+1))))
    dataframe = pandas.DataFrame()
    for i, container in enumerate(data_container_list):
        dataframe = pandas.concat([dataframe, pandas.DataFrame(
            container.md_to_list())], axis=1, ignore_index=True)
    dataframe.index = ['App_id', 'Label', 'Unit', 'Convergence_goal',
                       'Converged', 'Converged_run']+num_samples_index
    if out_format == 'csv':
        file_name = path+'.csv'
        dataframe.to_csv(file_name, index=True)
    elif out_format == 'hdf':
        file_name = path+'.h5'
        dataframe.to_hdf(file_name, key='df',
                         data_columns=data_header_cols, index=False)



def log_data(out_format, data_path_prefix, data_container_list):
    # Raggruppa i container per app_id
    apps_data = {}
    for container in data_container_list:
        if container.app_id not in apps_data:
            apps_data[container.app_id] = []
        apps_data[container.app_id].append(container)

    # Scrivi un file separato per ogni applicazione che ha raccolto dati
    for app_id, containers in apps_data.items():
        all_app_metrics = []

        for container in containers:
            # Se non ci sono dati o campioni, salta questo container
            if not container.data or not container.num_samples:
                continue

            # 1. Ricostruisci la colonna 'run_id' usando la lista num_samples
            run_ids = []
            for i, num in enumerate(container.num_samples):
                run_ids.extend([i + 1] * num) # i + 1 per avere run da 1, 2, 3...

            if len(run_ids) != len(container.data):
                print(f"[ATTENZIONE] Mismatch di dati per {container.get_title()}: "
                      f"{len(run_ids)} run ID generati, ma {len(container.data)} punti dati trovati. "
                      "Il logging per questa metrica potrebbe essere impreciso.")
                # Tronca la lista più lunga per evitare errori di DataFrame
                min_len = min(len(run_ids), len(container.data))
                run_ids = run_ids[:min_len]
                container.data = container.data[:min_len]

            # 2. Crea un DataFrame temporaneo per questa singola metrica
            metric_df = pandas.DataFrame({
                'run_id': run_ids,
                container.get_title(): container.data
            })
            
            # Imposta 'run_id' e un contatore progressivo come indice
            # per unire correttamente le diverse metriche
            metric_df = metric_df.set_index(['run_id', metric_df.groupby('run_id').cumcount()])
            all_app_metrics.append(metric_df)

        # 3. Unisce tutti i DataFrame delle metriche per questa app_id
        if not all_app_metrics:
            print(f"Nessun dato da salvare per App {app_id}.")
            continue
            
        dataframe = pandas.concat(all_app_metrics, axis=1).reset_index()
        # Rimuovi la colonna 'level_1' che viene creata da reset_index
        if 'level_1' in dataframe.columns:
            dataframe = dataframe.drop(columns=['level_1'])


        # Costruisci un nome di file specifico per questa app
        file_path = f"{data_path_prefix}_app_{app_id}"
        
        if out_format == 'csv':
            file_name = file_path + '.csv'
            # Salva il DataFrame finale che ora contiene la colonna 'run_id'
            dataframe.to_csv(file_name, index=False)
        elif out_format == 'hdf':
            file_name = file_path + '.h5'
            dataframe.to_hdf(file_name, key='df', index=False)
            
        print(f"Dati per App {app_id} salvati in: {file_name}")





def print_runtime(obj, mode, ro_file):
    if mode == 'stdout':
        print(obj)
    elif mode == 'none':
        return
    elif mode == '+file':
        print(obj)
        print(obj, file=ro_file)
    elif mode == 'file':
        print(obj, file=ro_file)


class Engine:
    def __init__(self, log_callback: Callable[[str], None] = print):
        self.log = log_callback

    def run(self, config: dict, environment: dict, is_worker: bool = False, output_dir: str = None):
        """
        Executes the benchmark.
        If is_worker is False, it acts as an orchestrator, creating and submitting a SLURM job.
        If is_worker is True, it runs the actual benchmark logic inside the SLURM allocation.
        """
        if is_worker:
            self._run_worker(config, environment, output_dir)
        else:
            self._run_orchestrator(config, environment)
    def _run_orchestrator(self, config: dict, environment: dict):
        """
        Orchestrator mode: Prepares directories, generates, and submits the SBATCH script.
        """
        self.log("Engine running in ORCHESTRATOR mode.")
        global_options = config.get('global_options', {})
        data_path = global_options.get('datapath', './data')
        extrainfo = global_options.get('extrainfo', '')
        num_nodes = int(global_options.get('numnodes'))
        ppn = int(global_options.get('ppn', 1))

        # --- Output Directory and Metadata Logging Setup ---
        try:
            self.log(f"[DEBUG] Checking/creating description file in '{data_path}'...")
            description_file = os.path.join(data_path, "description.csv")
            if not os.path.isfile(description_file):
                os.makedirs(data_path, exist_ok=True)
                with open(description_file, 'w') as desc_file:
                    desc_file.write('app_mix,system,numnodes,allocation_mode,allocation_split,ppn,out_format,extra,path\n')
            
            self.log("[DEBUG] Description file checked. Creating unique run directory...")
            
            # Aggiungiamo un timeout per sicurezza a questo ciclo
            timeout_start = time.time()
            while True:
                if time.time() - timeout_start > 10: # Timeout di 10 secondi
                    raise TimeoutError("Could not create a unique directory within 10 seconds. Check filesystem.")
                
                runner_id = (environment.get("CRAB_SYSTEM", "unknown") + "/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')) # Aggiungi microsecondi
                data_directory = os.path.join(data_path, runner_id)
                if not os.path.exists(data_directory):
                    os.makedirs(data_directory)
                    break
                time.sleep(0.01) # Piccolo sleep per evitare di ciclare troppo velocemente

            self.log(f"[DEBUG] Created run directory: {data_directory}")
            
            with open(description_file, 'a+') as desc_file:
                desc_line = f"{extrainfo},{environment.get('CRAB_SYSTEM')},{num_nodes},{global_options.get('allocationmode')},{global_options.get('allocationsplit')},{ppn},{global_options.get('outformat')},{extrainfo},{data_directory}\n"
                desc_file.write(desc_line)
            
            config_file_path = os.path.join(data_directory, 'config.json')
            with open(config_file_path, 'w') as config_file:
                json.dump(config, config_file, indent=4)

            # Salva l'ambiente processato in un file JSON che il worker leggerà.
            env_file_path = os.path.join(data_directory, 'environment.json')
            with open(env_file_path, 'w') as env_file:
                json.dump(environment, env_file, indent=4)
            self.log(f"[DEBUG] Saved processed environment to {env_file_path}")


            self.log("[DEBUG] Directory setup complete.")
        except Exception as e:
            self.log(f"[FATAL ERROR] Orchestrator failed during directory setup: {e}")
            raise # Rilancia l'eccezione per fermare tutto

        # --- SLURM Batch Script Generation ---
        sbatch_script_path = os.path.join(data_directory, 'crab_job.sh')
        python_executable = sys.executable
        cli_script_path = os.path.abspath(sys.argv[0])

        worker_command = (
            f"{python_executable} {cli_script_path} --worker "
            f"--workdir {data_directory}"
        )

        with open(sbatch_script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"#SBATCH --job-name=crab_{extrainfo[:10]}\n")
            f.write(f"#SBATCH --output={os.path.join(data_directory, 'slurm_output.log')}\n")
            f.write(f"#SBATCH --error={os.path.join(data_directory, 'slurm_error.log')}\n")
            f.write(f"#SBATCH --nodes={num_nodes}\n")
            f.write(f"#SBATCH --ntasks-per-node={ppn}\n")


            # For tests only
            #f.write(f"#SBATCH --exclusive\n")


            #TODO: rimettere l'if (per qualche motivo non funge)
            #if os.environ.get("CRAB_SYSTEM") == "leonardo":
                #TODO: far passare la partizione da config o env
                # f.write(f"#SBATCH --partition=boost_usr_prod\n")
                # f.write("#SBATCH --account=IscrB_SWING\n")
                # #TODO: capire in quali sistemi serve caricare i moduli, magari metterlo nell'env
                # f.write("module purge\n")
                # f.write("module load openmpi\n\n")
                # self.log("[DEBUG] Detected CRAB_SYSTEM=leonardo. Adding partition to SBATCH script.")

                #TODO: Capire il perche' di questi
                # f.write(f"#SBATCH --gres=tmpfs:0\n")
                # f.write(f"#SBATCH --time=01:00:00\n\n")



            venv_path = os.path.join(os.getcwd(), '.venv/bin/activate')
            f.write(f"source {venv_path}\n\n")

            f.write(f"{worker_command}\n")

        # --- Submit the Batch Script ---
        try:
            self.log(f"Submitting job with command: sbatch {sbatch_script_path}")
            submission_output = subprocess.check_output(['sbatch', sbatch_script_path], text=True)
            self.log(submission_output.strip())
            self.log(f"\nJob submitted. Monitor its status with 'squeue -u $USER'.")
            self.log(f"Results will be in: {data_directory}")
        except subprocess.CalledProcessError as e:
            self.log(f"Error submitting sbatch job: {e}")
            raise

    def _run_worker(self, config: dict, environment: dict, output_dir: str):
        """
        Worker mode: Executes the benchmark logic inside a SLURM allocation.
        """
        self.log("--- [WORKER] Engine running in WORKER mode. ---", flush=True)

        original_environ = os.environ.copy()
        try:
            os.environ.update(environment)

            # --- Inizializzazione del Worker ---
            pre_start_time = time.time()
            global_options = config.get('global_options', {})
            applications_config = config.get('applications', {})

            # La directory di lavoro è quella da cui viene eseguito lo script sbatch
            data_directory = output_dir # <-- USA QUESTA RIGA
            self.log(f"--- [WORKER] Working directory is: {data_directory} ---", flush=True)

            # --- Caricamento parametri dal config ---
            num_nodes = int(global_options.get('numnodes'))
            allocation_mode = global_options.get('allocationmode', 'l')
            allocation_split = global_options.get('allocationsplit', 'e')
            min_runs = int(global_options.get('minruns', 10))
            max_runs = int(global_options.get('maxruns', 1000))
            time_out = float(global_options.get('timeout', 100.0))
            ppn = int(global_options.get('ppn', 1))
            alpha = float(global_options.get('alpha', 0.05))
            beta = float(global_options.get('beta', 0.05))
            converge_all = bool(global_options.get('convergeall', False))
            out_format = global_options.get('outformat', 'csv')

            # --- Preparazione delle applicazioni (come prima) ---
            num_apps = 0
            apps = []
            apps_awaited = []
            apps_waiting = []
            schedule = []

            # (Il codice per inizializzare le app, schedule, ecc. è corretto e lo rimettiamo)
            sorted_apps = sorted(applications_config.items(), key=lambda item: int(item[0]))
            for app_key, app_details in sorted_apps:
                import_path_app = app_details.get("path")
                if not import_path_app: continue

                app_args = app_details.get("args", "")
                collect_flag = app_details.get("collect", False)
                start_val = app_details.get("start", "")
                end_val = app_details.get("end", "")

                start_time = 0.0 if start_val == '' else float(start_val)
                schedule.append((num_apps, 's', start_time))

                if end_val == '': apps_awaited.append(num_apps)
                elif end_val == 'f': apps_waiting.append(num_apps)
                else:
                    end_time = float(end_val)
                    schedule.append((num_apps, 't' if collect_flag else 'k', end_time))

                app_class_name = pathlib.Path(import_path_app).stem
                spec_app = importlib.util.spec_from_file_location(app_class_name, import_path_app)
                mod_app = importlib.util.module_from_spec(spec_app)
                spec_app.loader.exec_module(mod_app)
                apps.append(mod_app.app(num_apps, collect_flag, app_args))
                num_apps += 1

            # Ordina la schedule in base al tempo (terzo elemento della tupla)
            schedule.sort(key=lambda x: x[2])

            import_path_wlm = "./src/crab/core/wl_manager/" + os.environ["CRAB_WL_MANAGER"] + ".py"
            wlm_class_name = pathlib.Path(import_path_wlm).stem
            spec_wlm = importlib.util.spec_from_file_location(wlm_class_name, import_path_wlm)
            mod_wlm = importlib.util.module_from_spec(spec_wlm)
            spec_wlm.loader.exec_module(mod_wlm)
            wlmanager = mod_wlm.wl_manager()

            # --- Allocazione dei nodi ---
            self.log(f"--- [WORKER] Allocating nodes provided by SLURM: {os.environ.get('SLURM_NODELIST', 'N/A')} ---", flush=True)

            node_file_path = "worker_nodelist.txt"
            with open(node_file_path, "w") as f:
                subprocess.call(["scontrol", "show", "hostnames", os.environ['SLURM_NODELIST']], stdout=f)

            nodes_frame = pandas.read_csv(node_file_path, header=None)
            node_list = nodes_frame.iloc[:, 0].tolist()
            split_absolute = get_abs_split(allocation_split, num_apps, num_nodes)

            if allocation_mode == 'i': i_allocation(apps, node_list, split_absolute)
            else: l_allocation(apps, node_list, split_absolute)

            # --- Main Execution Loop (ora corretto) ---
            self.log('\nRunning Benchmarks...', flush=True)
            runs = 0
            converged = False
            start_time = time.time()
            data_container_list = [] # Inizializza qui
            for app in apps:
                if app.collect_flag:
                    for i in range(len(app.metadata)):
                        data_container_list.append(
                            data_container(app.id_num, app.metadata[i]["conv"], app.metadata[i]["name"], app.metadata[i]["unit"])
                        )

            while True:
                exec_time = time.time() - start_time
                if runs >= max_runs or (runs >= min_runs and converged) or exec_time >= time_out:
                    break


                self.log(f' Run {runs+1}:', flush=True)
                run_start_time = time.time()
                current_time = 0
                for app_id, action_type, action_time in schedule:
                    # Questa sleep ora sarà sempre non-negativa
                    time.sleep(action_time - current_time)
                    if action_type == 's':
                        run_job(apps[app_id], wlmanager, ppn)
                    else:
                        end_job(apps[app_id])
                    current_time = action_time

                for app_ind in apps_awaited:
                    wait_timed(apps[app_ind], time_out - (time.time() - start_time))
                for app_ind in apps_waiting:
                    end_job(apps[app_ind])

                runs += 1

                # Controlla i codici di ritorno, ma ignora gli errori per i job
                # che sono stati terminati forzatamente di proposito.
                for app in apps:
                    # Se l'applicazione non ha un processo, salta.
                    if not hasattr(app, 'process'):
                        continue
                    
                    # Se il processo è stato terminato forzatamente da noi ('end': 'f'),
                    # un codice di uscita non-zero (come -9 per SIGKILL) è atteso e OK.
                    # apps_waiting contiene gli indici delle app con 'end':'f'
                    if app.id_num in apps_waiting and app.process.returncode != 0:
                        self.log(f"[INFO] Application {app.id_num} was intentionally terminated (exit code: {app.process.returncode}). This is expected.", flush=True)
                        continue # Passa alla prossima app senza generare errore

                    # Per tutte le altre applicazioni, un codice non-zero è un errore reale.
                    # if app.process.returncode not in [None, 0]:
                        # Pulisci stderr per renderlo più leggibile
                        # stderr_clean = app.stderr.strip() if app.stderr else "No stderr."
                        # raise Exception(f"Application {app.id_num} failed with exit code {app.process.returncode}. Stderr: {stderr_clean}")




                container_idx = 0
                for app in apps:
                    if app.collect_flag and hasattr(app, 'process') and app.process.returncode == 0:
                        data_list_of_list = app.read_data()
                        for data_list in data_list_of_list:
                            data_container_list[container_idx].data.extend(data_list)
                            data_container_list[container_idx].num_samples.append(len(data_list))
                            container_idx += 1

                if runs >= min_runs:
                    converged = check_CI(data_container_list, alpha, beta, converge_all, runs)

                run_end_time = time.time()
                run_duration = run_end_time - run_start_time
                self.log(f"--- [INFO] Run {runs} completata in {run_duration:.4f} secondi ---", flush=True)


            # --- Final Data Logging  ---
            self.log('\nLogging data...', flush=True)
            if data_container_list:
                log_data(out_format, os.path.join(data_directory, 'data'), data_container_list)
                #log_meta_data(out_format, os.path.join(data_directory, 'metadata'), data_container_list, runs)


        finally:
            os.environ.clear()
            os.environ.update(original_environ)
            if os.path.exists(node_file_path):
                os.remove(node_file_path)