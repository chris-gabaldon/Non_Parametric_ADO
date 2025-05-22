# experimento/views.py

from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import ExperimentSession, Block, Trial, ExperimentManager
import numpy as np
import random
from django.utils import timezone
from experimento.utils import generate_mean_list
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from experimento.utils import generate_mean_list, Experiment, Prior
import json
import random
# experimento/views.py

import json


N=25 # numero total de trials
manager = ExperimentManager()

# @csrf_exempt
@csrf_exempt  # Remove this in production if using CSRF protection
def end_experiment(request):

    if request.method == 'POST':
        try:
            # Parsear la solicitud JSON
            data = json.loads(request.body)
            # Extraer los datos enviados desde el frontend
            session_id = data.get('session_id')
            correct_responses = data.get('correct_responses')
            email = data.get('email')  # Correo electrónico
            age = data.get('age')  # Edad
            gender = data.get('gender')  # Género
            education = data.get('education')  # Nivel educativo

            # Buscar la sesión del experimento
            session = ExperimentSession.objects.filter(id=session_id).first()
            
            if session:
                # Actualizar los datos en la sesión
                session.correct_responses = correct_responses
                session.email = email
                session.age = age  # Nuevo campo
                session.gender = gender  # Nuevo campo
                session.education = education  # Nuevo campo
              # Actualizar la hora de finalización
                session.experiment_data = None
                session.save()

                return JsonResponse({'status': 'success', 'message': 'Resultados guardados correctamente'})
            else:
                return JsonResponse({'status': 'error', 'message': 'Sesión no encontrada'})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Método no permitido'})


def save_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        #print('save_response', data)

        trial_id = data.get('trial_id')  # This can be used as trial_number
        response = data.get('response')
        ExperimentSession.last_response = response 
        reaction_time = data.get('reaction_time')
        session = ExperimentSession.objects.filter(id=session_id).first()
        print(session.num_list, "lista acá")
        stimulus = session.num_list[-2]  # New data to create a trial
        block_id = 1 # data.get('block_id')  # Assuming you send the block ID in the request
        part_id = data.get('participante')
        ado_type=data.get('ado_type')
        session = ExperimentSession.objects.get(id=part_id)
        participant_id = session.participant_id
        #print("aca", participant_id)
        d1=data.get('d1')
        
        
        
        #print(part_id)

        try:
            # Retrieve the Block instance using the provided block_id
            block = Block.objects.get(id=block_id)
            
            # Create a new Trial object with the Block instance
            trial = Trial(
                block=block,  # Assign the Block instance here
                trial_id=trial_id,
                stimulus=stimulus,
                response=response,
                # correct=False,  # Default value, will set based on logic below
                reaction_time=reaction_time,
                participant_id = participant_id,
                d1=d1,
                ado_type=ado_type
                
                
            )
   
            trial.save()
            #if (stimulus <= 50 and response == "<") or (stimulus >= 50 and response == ">"):
            #corr_esp_trial = 1
        #else:
            #corr_esp_trial = 0
            
            return JsonResponse({'status': 'success', 'trial_id': trial.id})  # Optionally return the new trial ID
        except Block.DoesNotExist:
            return JsonResponse({'status': 'fail', 'message': 'Block not found'}, status=404)
        except Exception as e:
            print(f"Error creating trial: {e}")
            return JsonResponse({'status': 'fail', 'message': 'Error creating trial'}, status=500)

    return JsonResponse({'status': 'fail'}, status=400)


def index(request):
    if request.method == 'POST':
        participant_id_random = random.randint(1,1000000)
        #p_random = Trial.object.create(participante_id_random = participant_id_random)
        session = ExperimentSession.objects.create(participant_id=participant_id_random)

        return redirect('start_experiment', session_id=session.id)
    return render(request, 'experimento/index.html')

def start_experiment(request, session_id):
    session = ExperimentSession.objects.get(id=session_id)

    return render(request, 'experimento/start_experiment.html', {'session_id': session.id})

def run_block(request, session_id, block_number):
    session = ExperimentSession.objects.get(id=session_id)
    block = Block.objects.create(session=session, block_number=block_number)
    participant_id = session.participant_id
    #print(participant_id)
    # Datos iniciales del primer ensayo
    d = [41, 44, 46, 48, 49, 51, 52, 54, 56, 59]
    prom_fijo = 50
    std = 15
    large = 8
    D=len(d)
    k=100
    prior = Prior() 
    
    exp = manager.get_experiment(0)
    
    prior_type=random.choice([1,2]) #1: gauss  2: ni
    session.prior_type = prior_type 
    session.save()   

    if prior_type ==1:
        exp.generate(D,k)
        p = prior.set_prior_gauss(d[-1],D,k) # calcualte the prior based on exponential curves, shown in Figure 1b
        exp.set_prior(p)
        print('GAUSSIAN prior')
    else:
        print('Non Informative prior')
        exp.generate(D,k)

    
    # Genera el primer ensayo
    
    ado_type=random.choice([20]) # para que haya una proporcion  uniforme de: random_(cualquier_prior), ado_ni,ado_gauss
    print('ado_type',ado_type)
    if ado_type ==10:
        d1 = 4
        print('ADO')
    if ado_type==20:
        d1 = 4
        print('Random')
    print(d1, "D1")
    numeros = generate_mean_list(d[d1], std, large)
    numeros = [int(n) for n in numeros]
    print(numeros, "num ok")
    session.experiment_data = manager.serialize_experiment() 
    session.num_list.append(numeros)
    session.save()
    trial_data = {
        'trial_number': 1,
        'stimulus': numeros,
        'd1': d1,
        'd_value': d[d1],
        'ado_type':ado_type,

    }
    # print(trial_data)
    return render(request, 'experimento/run_block.html', {
        'session_id': session.id,
        'block_id': block.id,
        'trial_data': trial_data, 
        'participant_id': participant_id
    })

def generate_trial_data(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        d1_viejo= data.get('d1')
        response = data.get('response')
        trial_number = data.get('trial_number') + 1  # Incrementar número de ensayo
        session_id = data.get('session_id')
        trial_id = data.get('trial_id')

        ado_type=data.get('ado_type')
        session = ExperimentSession.objects.get(id=session_id)
        print(ado_type)
        
        # Imprimir la respuesta del ensayo actual antes de generar la nueva secuencia de números
        # print(f"Respuesta del ensayo {trial_number - 1}: {response}, ")
        
        # Generar nueva secuencia de estímulos para el siguiente ensayo
        d = [41, 44, 46, 48, 49, 51, 52, 54, 56, 59]
        D=len(d)
        prom_fijo = 50
        std = 15
        large = 8
        if response == "<":
            y=0
        if response =='>':
            y=1
        #print('d=',d1_viejo,'y=',y,'ado_type',ado_type)
        session = ExperimentSession.objects.get(id=session_id) 
        if session.experiment_data:
             manager.deserialize_experiment(session.experiment_data) 
        exp = manager.get_experiment(1)
        print(trial_number,'trial number')
        exp.update(d1_viejo,y)
        
        if trial_number== 2:
            n_first_loops=18
            random_trials=[range(D)[mm % D] for mm in range(n_first_loops*D)]
            random.shuffle(random_trials)
            random_trials.append(4)
            session.random_trials=random_trials
            ff = [5, 4, 6, 5, 2, 1, 3, 10,11,10,11,6, 0, 1, 8, 9, 2, 7, 0, 7, 9, 3, 8]
            # random.shuffle(ff)
            session.ff = ff 
            session.save()#-->lo guardo en el modelo
            d1=ff[0]
            print('fff',ff,d1,'D1')
        elif trial_number<25:
            ff = session.ff # Traer ff del modelo 
            d1 = ff[trial_number-2]
            print(d1,ff,'elif')
        else:       
            if ado_type ==10:
                d1 = exp.ADOchoose()
                print('ADO',ado_type)
            if ado_type==20: 
                random_trials=session.random_trials
                d1 = random_trials[trial_number-25]
                print('random',ado_type)
        d1=int(d1)
        print(d1,'D1')
        numeros = generate_mean_list(d[d1], std, large)
        numeros = [int(n) for n in numeros]
        session = ExperimentSession.objects.filter(id=session_id).first()
        # print(numeros, "numeros")
        # print(session.num_list, "probando")
        # print(session.num_list[-1], response)
        # print(sum(session.num_list[-1])/len(session.num_list[0]), "promedios")   #promedio
        prom = sum(session.num_list[-1])/len(session.num_list[0])
        

        if session:
                # Verificar la condición y sumar 1 a corr_trial si se cumple
            if (prom <= 50 and response == "<") or (prom >= 50 and response == ">"):
                session.corr_trial += 1
                
            session.save()


        session.num_list.append(numeros)
        if len(session.num_list) > 3:
            print("entrando")
            session.num_list.pop(0)
            print(session.num_list, "lista")
        session.experiment_data = manager.serialize_experiment()    
        session.save()
        # Preparar datos para el nuevo ensayo
        trial_data = {
            'trial_number': trial_number,
            'stimulus': numeros,
            'd1': d1,
            'd_value': d[d1],
            'prom_fijo': prom_fijo,
            'y':y,
            'ado_type':ado_type,
        }
        # print(trial_data)
        return JsonResponse({'status': 'success', 'trial_data': trial_data})



def finish_experiment(request, session_id):
    session = ExperimentSession.objects.get(id=session_id)
    session.end_time = timezone.now()
    session.save()
    corr_trial = session.corr_trial
    return render(request, 'experimento/finish_experiment.html', {'corr_trial': corr_trial})
