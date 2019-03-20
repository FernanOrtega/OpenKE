import config
import models
import os
from time import time
import json


def train_and_get_embeddings(root_path: str, dataset: str, site: str, params: dict, model: models.Model) -> []:
    # Input training files from benchmarks/FB15K/ folder.
    con = config.Config()
    # True: Input test files from the same folder.
    con.set_in_path(os.path.join(root_path, dataset, site, ''))
    con.set_work_threads(params.get('work_threads'))
    con.set_train_times(params.get('train_times'))
    con.set_nbatches(params.get('nbatches'))
    con.set_alpha(params.get('alpha'))
    con.set_margin(params.get('margin'))
    con.set_bern(params.get('bern'))
    con.set_dimension(params.get('dimension'))
    con.set_ent_neg_rate(params.get('ent_neg_rate'))
    con.set_rel_neg_rate(params.get('rel_neg_rate'))
    con.set_opt_method(params.get('opt_method'))

    results_path = os.path.join(params.get('results_base'), model.__name__,
                                dataset, site)
    os.makedirs(results_path, exist_ok=True)
    # Models will be exported via tf.Saver() automatically.
    con.set_export_files(os.path.join(results_path, "model.vec.tf"), 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files(os.path.join(results_path, "embedding.vec.json"))
    # Initialize experimental settings.
    con.init()
    # Set the knowledge embedding model
    con.set_model(model)
    # Train the model.
    con.run()

    return con.get_parameters()


def split_embeddings(embedding: {}, target_path: str, idx2entity: {}, idx2clazz: {}) -> None:
    a_entity_embeddings = embedding.get('ent_embeddings') if 'ent_embeddings' in embedding \
        else embedding.get('ent_re_embeddings')
#       else embedding.get('ent_im_embeddings')
    os.makedirs(target_path, exist_ok=True)
    training_csv_path = os.path.join(target_path, 'training.csv')
    testing_csv_path = os.path.join(target_path, 'testing.csv')

    with open(training_csv_path, 'w') as training_embeddings:
        with open(testing_csv_path, 'w') as testing_embeddings:
            header = ','.join(['node','clazz'] + ['f'+str(position).zfill(3)
                                                  for position in range(params.get('dimension'))]) + '\n'
            training_embeddings.write(header)            
            testing_embeddings.write(header)

            for idx, entity_embeddings in enumerate(a_entity_embeddings):
                entity = idx2entity.get(str(idx))
                clazz_vector = idx2clazz.get(str(idx))
                clazz = clazz_vector[1]
                is_training = clazz_vector[0]
                line = ','.join([entity] + [clazz] + [str(x) for x in entity_embeddings]) + '\n'
                if is_training == 'true':
                    training_embeddings.write(line)
                elif is_training == 'false':
                    testing_embeddings.write(line)
                else: # For null cases, we ignore them.
                    pass


def get_idx2entity(domain_path: str, site: str) -> {}:
    result = {}
    file_path = os.path.join(domain_path, site, 'entity2id.txt')
    for line in [line.rstrip('\n') for line in open(file_path)][1:]:
        values = line.split()
        result[values[1]] = values[0]

    return result


def get_idx2clazz(domain_path: str, site: str) -> {}:
    result = {}
    clazz_path = os.path.join(domain_path, site, 'clazzes.txt')
    for line in [line.rstrip('\n') for line in open(clazz_path)][1:]:
        values = line.split()
        result[values[0]] = values[1:]

    return result


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root_path = 'KGEAquilaDatasets'
embeddings_path = './KGEEmbeddings'
params = {
    'work_threads': 8,
    'train_times': 600,
    'nbatches': 100,
    'alpha': 0.001,
    'margin': 1.0,
    'bern': 0,
    'dimension': 100,
    'ent_neg_rate': 1,
    'rel_neg_rate': 0,
    'opt_method': 'SGD',
    'results_base': './KGEModel'
}
models_list = [models.ComplEx]
""" models_list = [
    models.Analogy, models.ComplEx, models.DistMult, models.HolE,
    models.RESCAL, models.TransD, models.TransE, models.TransH, models.TransR
    ] """

if os._exists(embeddings_path):
    os.removedirs(embeddings_path)
if os._exists(params.get('results_base')):
    os.removedirs(params.get('results_base'))

times = {}
for model in models_list:
    times[model.__name__] = {}
    for domain in os.listdir(root_path):
        times[model.__name__][domain] = {}
        domain_path = os.path.join(root_path, domain)
        for site in os.listdir(domain_path):
            print('Training dataset {} for site {}.'.format(domain, site))
            t_init = time()
            embeddings = train_and_get_embeddings(root_path, domain, site,
                                                  params, model)
            t_end = time()
            t_total = t_end - t_init
            print('Training time: {}'.format(t_total))
            split_embeddings(embeddings, os.path.join(embeddings_path, model.__name__, domain,
                             site), get_idx2entity(domain_path, site),
                             get_idx2clazz(domain_path, site))
            times[model.__name__][domain][site] = t_total

json = json.dumps(times)
with open("times.json", "w") as f:
    f.write(json)
