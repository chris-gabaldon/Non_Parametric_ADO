from django.contrib import admin
from .models import Trial,Block, ExperimentSession

class ExperimentSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'participant_id', 'email', 'prior_type', 'start_time', 'end_time', 'correct_responses')  # Agregar prior_type
    search_fields = ('email', 'participant_id')  # Búsqueda por email y participant_id
    list_filter = ('start_time', 'end_time', 'prior_type')  # Filtrar por prior_type

# Registra el modelo para que sea visible en la administración
admin.site.register(Trial)
admin.site.register(Block)
admin.site.register(ExperimentSession, ExperimentSessionAdmin)

