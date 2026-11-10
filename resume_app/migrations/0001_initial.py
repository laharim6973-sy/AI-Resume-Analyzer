from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='ResumeAnalysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_name', models.CharField(max_length=255)),
                ('file_path', models.FileField(blank=True, null=True, upload_to='resumes/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(
                    choices=[('pending', 'Pending'), ('processing', 'Processing'),
                             ('completed', 'Completed'), ('failed', 'Failed')],
                    default='pending', max_length=20)),
                ('raw_text', models.TextField(blank=True)),
                ('ats_score', models.FloatField(default=0)),
                ('skills_json', models.TextField(blank=True, default='{}')),
                ('sections_json', models.TextField(blank=True, default='{}')),
                ('contact_json', models.TextField(blank=True, default='{}')),
                ('recommendations_json', models.TextField(blank=True, default='[]')),
                ('predicted_roles_json', models.TextField(blank=True, default='[]')),
                ('stats_json', models.TextField(blank=True, default='{}')),
                ('jd_match_json', models.TextField(blank=True, default='{}')),
                ('experience_years', models.FloatField(default=0)),
                ('job_description', models.TextField(blank=True)),
                ('error_message', models.TextField(blank=True)),
            ],
            options={
                'verbose_name': 'Resume Analysis',
                'verbose_name_plural': 'Resume Analyses',
                'ordering': ['-uploaded_at'],
            },
        ),
    ]
