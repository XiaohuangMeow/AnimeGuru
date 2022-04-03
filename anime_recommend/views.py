import re
import os
import pylab as plt
import numpy as np
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render, redirect
from .models import Anime, Myrating, MyList
from .forms import UserRegisterForm
from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Case, When
from .recommend import *
# Create your views here.


def info(request):
    return render(request, 'anime_recommend/home.html')

def paginate(animes, cur_page, num_per_page):
    paginator = Paginator(animes, num_per_page)

    try:
        animes = paginator.page(cur_page)
    except PageNotAnInteger:
        animes = paginator.page(1)
    except EmptyPage:
        animes = paginator.page(paginator.num_pages)
    return animes 


def index(request):
    # fetch all the animes stored in the database
    myrequest = dict()
    if request.method == 'GET':
        myrequest['StartYear'] = request.GET.get('StartYear', None)
        myrequest['Tags'] = request.GET.get('Tags', None)
        myrequest['Finished'] = request.GET.get('Finished', None)
        myrequest['Content Warning'] = request.GET.get('Content_Waring', None)
        myrequest['Episodes'] = request.GET.get('Episodes', None)
        myrequest['sort'] = request.GET.get('sort', None)

        page = request.GET.get('page', 1)
        
        animes = list_filter(myrequest)
        animes = paginate(animes, page, 24)
    else:
        print(request.POST)
        myrequest['StartYear'] = request.POST.get('StartYear', None)
        myrequest['Tags'] = request.POST.get('Tags', None)
        myrequest['Finished'] = request.POST.get('Finished', None)
        myrequest['Content Warning'] = request.POST.get('Content_Waring', None)
        myrequest['Episodes'] = request.POST.get('Episodes', None)
        myrequest['sort'] = request.POST.get('sort', None)

        page = request.POST.get('page', 1)

        animes = list_filter(myrequest)

    if request.user.is_authenticated:
        messages.success(
            request, f'Now click profile to see your watch list')
        messages.success(
            request, f'Now click Get Recommendations to tell us your preference')
    else:
        messages.success(
            request, f'Register/Log in to enjopy our anime recommendation service!')

    return render(request, 'anime_recommend/list.html', {'animes': animes, 'myrequest':myrequest})

# Register user
def register(request):
    # if user have clicked the submitted for signning up 
    if request.method == "POST":

        form = UserRegisterForm(request.POST)
        # check if the form is valid
        if form.is_valid():
            
            # save this user created from this form
            form.save()
            # get user name
            username = form.cleaned_data['username']
            # display a flash message
            messages.success(request, f'Account created for {username}! You are now able to login')
            return redirect("login")
    else:
        form = UserRegisterForm()
    

    context = {'form': form}

    return render(request, 'anime_recommend/register.html', context)

# helper function to convert data in db to dataframe
def get_animes_df():
    animes_df = pd.DataFrame(list(Anime.objects.all().values()))

    animes_df = animes_df.rename(
        columns={'Anime_PlanetID': 'Anime-PlanetID', 
                 'Alternative_Name': 'Alternative Name',
                 'Rating_score': 'Rating Score',
                 'Number_votes': 'Number Votes',
                 'Content_Warning': 'Content Warning'})

    animes_df['Anime-PlanetID'] = animes_df['Anime-PlanetID'].astype(np.int64)
    animes_df['Rating Score'] = animes_df['Rating Score'].astype(np.float64)
    animes_df['Number Votes'] = animes_df['Number Votes'].astype(np.int64)
    animes_df['Duration'] = animes_df['Duration'].astype(np.int64)
    animes_df['StartYear'] = animes_df['StartYear'].astype(np.int64)
    animes_df['EndYear'] = animes_df['EndYear'].astype(np.int64)
    animes_df['Finished'] = animes_df['Finished'].map({'False': False, 'True': True})
    animes_df['Finished'] = animes_df['Finished'].astype(np.bool)

    return animes_df

def list_filter(myrequest):
    animes_df = get_animes_df()
    _, animes_selected = anime_filter(animes_df, myrequest)


    animes_selected_idx = animes_selected['Anime-PlanetID'].tolist()
    preserved = Case(*[When(Anime_PlanetID=Anime_PlanetID, then=pos)
                       for pos, Anime_PlanetID in enumerate(animes_selected_idx)])

    anime_list = list(Anime.objects.filter(
        Anime_PlanetID__in=animes_selected_idx).order_by(preserved))


    return anime_list

@login_required
def profile(request):
    animes = Anime.objects.filter(mylist__watch=True, mylist__user=request.user)

    return render(request, 'anime_recommend/profile.html', {'animes': animes})

@login_required
def preference_recommend(request):
    animes_df = get_animes_df()

    myrequest = dict()
    if request.method == 'GET':
        return render(request, 'anime_recommend/recommend.html')

    print(request.POST)
    myrequest['StartYear'] = request.POST.get('StartYear', None)
    if myrequest['StartYear'] == 'before2015':
        myrequest['StartYear'] = 2016
    else:
        myrequest['StartYear'] = int(myrequest['StartYear'])

    myrequest['Finished'] = request.POST.get('Finished', None)
    if myrequest['Finished'] == '' or myrequest == 'True':
        myrequest['Finished'] = True
    else:
        myrequest['Finished'] = False
    
    myrequest['Type'] = request.POST.get('Type', None)
    myrequest['Tags'] = request.POST.get('Tags', None)
    myrequest['Episodes'] = request.POST.get('Episodes', None)
    myrequest['Duration'] = request.POST.get('Duration', None)
    
    page = request.POST.get('page', 1)
    preference = pd.DataFrame(myrequest, index=[0])

    recommend_anime, _ = recommender(animes_df, preference)

    animes_selected_idx = recommend_anime['Anime-PlanetID'].tolist()
    preserved = Case(*[When(Anime_PlanetID=Anime_PlanetID, then=pos)
                       for pos, Anime_PlanetID in enumerate(animes_selected_idx)])

    anime_list = list(Anime.objects.filter(
        Anime_PlanetID__in=animes_selected_idx).order_by(preserved))

    return render(request, 'anime_recommend/recommend.html', {'animes': anime_list, 'myrequest': request.POST})

@login_required
def detail(request, anime_id):
    anime = Anime.objects.get(id=anime_id)
    user_animes = list(MyList.objects.all().values().filter(
        anime_id=anime_id, user=request.user))
    if user_animes:
        watched = user_animes[0]['watch']
    else:
        watched = False

    if request.method == 'POST':
        watch_values = request.POST['watch']
        print(request.POST)
        if 'Remove' in watch_values:
            watched = False
            messages.success(request, "This anime is removed from your list!")

        if 'on' in watch_values:
            watched = True
            messages.success(request, "Anime added!")

        if user_animes:
            MyList.objects.all().values().filter(anime_id=anime_id,user=request.user).update(watch=watched)
        else:
            q=MyList(user=request.user,anime=anime,watch=watched)
            q.save()

    context = {'anime':anime, 'watch':watched}
    print(context)
    return render(request, 'anime_recommend/detail.html', context)
