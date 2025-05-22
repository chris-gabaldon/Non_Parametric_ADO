
from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('generate_trial_data/', views.generate_trial_data, name='generate_trial_data'),
    path('start/<int:session_id>/', views.start_experiment, name='start_experiment'),
    path('block/<int:session_id>/<int:block_number>/', views.run_block, name='run_block'),
    path('save-response/', views.save_response, name='save_response'),
    path('finish/<int:session_id>/', views.finish_experiment, name='finish_experiment'),
    path('end-experiment/', views.end_experiment, name='end_experiment'),
]
