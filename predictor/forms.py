from django import forms

AGE_CHOICES = [('18_20', '18-20'), ('21_24', '21-24'), ('25_29', '25-29'), ('30_35', '30-35')]
EDUCATION_CHOICES = [('someSecondary', 'Some Secondary'), ('ordinaryLevel', 'Ordinary Level'),
                      ('aLevel', 'A Level'), ('tetiary', 'Tertiary')]
INCOME_CHOICES = [('100-', 'Under 100'), ('101_250', '101-250'), ('251_500', '251-500'), ('500+', '500+')]
YES_NO_CHOICES = [('yes', 'Yes'), ('no', 'No')]
FREQ_CHOICES = [('never', 'Never'), ('rarely', 'Rarely'), ('sometimes', 'Sometimes'),
                 ('often', 'Often'), ('always', 'Always')]
LIKERT_CHOICES = [('stronglyDisagree', 'Strongly Disagree'), ('disagree', 'Disagree'),
                   ('neutral', 'Neutral'), ('agree', 'Agree'), ('stronglyAgree', 'Strongly Agree')]
FRIENDS_USE_CHOICES = [('0', '0'), ('1_2', '1-2'), ('3_4', '3-4'), ('5', '5')]
APPROVAL_CHOICES = [('stronglyDisapprove', 'Strongly Disapprove'), ('disapprove', 'Disapprove'),
                     ('neutral', 'Neutral'), ('approve', 'Approve'), ('stronglyApprove', 'Strongly Approve')]
COUNT_CHOICES = [('0', '0'), ('1_2', '1-2'), ('3_4', '3-4'), ('5+', '5+')]
HOURS_CHOICES = [('1-', 'Under 1'), ('1_3', '1-3'), ('3_5', '3-5'), ('5+', '5+')]
EXPOSURE_CHOICES = [('never', 'Never'), ('rarely', 'Rarely'), ('sometimes', 'Sometimes'),
                     ('often', 'Often'), ('daily', 'Daily')]

GENDER_CHOICES = [('male', 'Male'), ('female', 'Female')]
PROVINCE_CHOICES = [('harare', 'Harare'), ('manicaland', 'Manicaland'), ('mashEast', 'Mashonaland East'),
                     ('midlands', 'Midlands'), ('other', 'Other')]
ACTIVITY_CHOICES = [('student', 'Student'), ('employed', 'Employed'), ('selfEmployed', 'Self-Employed'),
                     ('pieceJob', 'Piece Job'), ('unemployedSeekingWork', 'Unemployed, Seeking Work'),
                     ('neet', 'Not in Education, Employment or Training')]
EMPLOY_STATUS_CHOICES = [('employed', 'Employed'), ('selfEmployed', 'Self-Employed'),
                          ('unemployed', 'Unemployed'), ('retired', 'Retired'),
                          ('preferNotToSay', 'Prefer Not to Say')]
LIVING_CHOICES = [('bothBiologicalParents', 'Both Biological Parents'),
                   ('livingWithSingleParent', 'Single Parent'),
                   ('livingWithGuardian', 'Guardian'), ('other', 'Other')]
DIASPORA_CHOICES = [('notApplicable', 'Not Applicable'), ('noLivedWithMe', 'No, Lived With Me'),
                     ('yesLeftWithGuardian', 'Yes, Left With Guardian'),
                     ('yesLeftWithOtherParent', 'Yes, Left With Other Parent')]
RELATIVE_DRUGS_CHOICES = [('never', 'Never'), ('yesInThePast', 'Yes, In The Past'),
                           ('yesCurrently', 'Yes, Currently'), ('preferNotToSay', 'Prefer Not to Say')]
PLATFORM_CHOICES = [('facebook', 'Facebook'), ('instagram', 'Instagram'), ('whatsapp', 'WhatsApp'),
                     ('tiktok', 'TikTok'), ('telegram', 'Telegram'), ('youtube', 'YouTube'), ('other', 'Other')]


class AssessmentForm(forms.Form):
    age = forms.ChoiceField(choices=AGE_CHOICES, label="Age group")
    gender = forms.ChoiceField(choices=GENDER_CHOICES)
    province = forms.ChoiceField(choices=PROVINCE_CHOICES)
    highestEducation = forms.ChoiceField(choices=EDUCATION_CHOICES, label="Highest education")
    main_activity = forms.ChoiceField(choices=ACTIVITY_CHOICES, label="Main activity")
    monthlyIncomeTotal = forms.ChoiceField(choices=INCOME_CHOICES, label="Monthly household income")
    employStatusHead = forms.ChoiceField(choices=EMPLOY_STATUS_CHOICES, label="Household head's employment status")
    strugle12MonthsHead = forms.ChoiceField(choices=FREQ_CHOICES, label="Household struggled financially (past 12 months)")
    livingSitu18Years = forms.ChoiceField(choices=LIVING_CHOICES, label="Living situation up to age 18")
    parentsWorkedDiaspora6Months = forms.ChoiceField(choices=DIASPORA_CHOICES, label="Parent worked abroad 6+ months")
    relativeProblemWithDrugs = forms.ChoiceField(choices=RELATIVE_DRUGS_CHOICES, label="Relative with a drug problem")

    comfortableSharingPersonalProblems = forms.ChoiceField(choices=LIKERT_CHOICES, label="Comfortable sharing personal problems")
    eatDinnerWithFamily = forms.ChoiceField(choices=LIKERT_CHOICES, label="Regularly eats dinner with family")
    guadianKnowMyWhereabouts = forms.ChoiceField(choices=LIKERT_CHOICES, label="Guardian knows my whereabouts")

    outOf5FriendsUseDrugs = forms.ChoiceField(choices=FRIENDS_USE_CHOICES, label="Out of 5 friends, how many use drugs")
    howFriendsFeelTryingDrugs = forms.ChoiceField(choices=APPROVAL_CHOICES, label="How friends feel about trying drugs")
    unsupervisedEveningsInAWeek = forms.ChoiceField(choices=COUNT_CHOICES, label="Unsupervised evenings per week")
    participationInOrganizedProsocialActivitiesInAWeek = forms.ChoiceField(choices=COUNT_CHOICES, label="Organized prosocial activities per week")
    hoursSpentOnSocialMedia = forms.ChoiceField(choices=HOURS_CHOICES, label="Hours spent on social media per day")
    mostUsedPlatform = forms.ChoiceField(choices=PLATFORM_CHOICES, label="Most used platform")
    howTimesSeenDrugSupportContentIn30Days = forms.ChoiceField(choices=EXPOSURE_CHOICES, label="Seen drug support content (last 30 days)")