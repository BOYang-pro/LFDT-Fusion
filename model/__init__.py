import logging
logger = logging.getLogger('base')


def create_model(opt,local_rank):
    from model.model import Diffusion_Fusion_Model
    diff_model = Diffusion_Fusion_Model(opt,local_rank)
    logger.info('Model [{:s}] is created.'.format(diff_model.__class__.__name__))
    return diff_model