FROM jupyter/scipy-notebook

RUN pip install --quiet \
    'tensorflow==1.15' gpt-2-simple && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
