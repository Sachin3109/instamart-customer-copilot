import subprocess,sys
for s in ['generate_data.py','part_a.py','part_b.py','part_c.py','part_d.py']:
 print('\n###',s,flush=True); subprocess.run([sys.executable,s],check=True)
