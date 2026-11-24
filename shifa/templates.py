"""Synthetic message templates for Shifa (no real patient data, PDPL).

TEMPLATES[topic][style] = 14 (intent, text) pairs.
Indices 0-7 are used for training, 8-10 for validation, 11-13 for test: validation and test
contain WORDINGS the model never saw, so the model has to generalise, not memorise.
Each split holds every intent of the topic, and every split's templates include entity-free wordings.
Every {slot} must be a whole whitespace-separated token (alignment relies on it).
"""

SLOT_ENTITY = {"doc": "PERSON", "date": "DATE", "dept": "DEPT", "drug": "DRUG", "test": "TEST", "amt": "MONEY"}

SLOT_VALUES = {
    "gulf": {
        "doc": ["د. خالد العتيبي", "د. سارة الغامدي", "د. فهد الشمري", "د. نورة القحطاني", "د. محمد الدوسري"],
        "date": ["بكرة", "يوم الأحد", "الخميس الجاي", "الأسبوع الجاي", "بعد بكرة", "يوم الثلاثاء"],
        "dept": ["العيون", "الأسنان", "الباطنية", "العظام", "الجلدية", "الأطفال", "القلب"],
        "drug": ["بنادول", "اموكسيسيلين", "ميتفورمين", "اوميبرازول", "فينتولين"],
        "test": ["السكر التراكمي", "صورة الدم", "فحص الكوليسترول", "وظائف الكلى"],
        "amt": ["350 ريال", "1200 ريال", "90 ريال", "480 ريال"],
    },
    "msa": {
        "doc": ["الدكتور خالد العتيبي", "الدكتورة سارة الغامدي", "الدكتور فهد الشمري", "الدكتورة نورة القحطاني"],
        "date": ["غدا", "يوم الأحد", "الخميس القادم", "الأسبوع القادم", "الثلاثاء الماضي", "السبت الماضي"],
        "dept": ["العيون", "الأسنان", "الباطنية", "العظام", "الجلدية", "الأطفال", "القلب"],
        "drug": ["بنادول", "أموكسيسيلين", "ميتفورمين", "أوميبرازول", "فنتولين"],
        "test": ["السكر التراكمي", "صورة الدم الكاملة", "الكوليسترول", "وظائف الكلى"],
        "amt": ["350 ريالا", "1200 ريال", "90 ريالا", "480 ريال"],
    },
    "en": {
        "doc": ["Dr. Sara Alghamdi", "Dr. Khalid Alotaibi", "Dr. Fahad Alshammari", "Dr. Noura Alqahtani"],
        "date": ["tomorrow", "next Sunday", "on Thursday", "12 March", "next week", "on 3 May"],
        "dept": ["Cardiology", "Dermatology", "Orthopedics", "Pediatrics", "Dentistry", "Ophthalmology"],
        "drug": ["Panadol", "Amoxicillin", "Metformin", "Omeprazole", "Ventolin"],
        "test": ["HbA1c", "CBC", "cholesterol panel", "kidney function test"],
        "amt": ["SAR 350", "SAR 1200", "SAR 90", "SAR 480"],
    },
}

GREETINGS = {
    "gulf": ["السلام عليكم", "هلا", "مرحبا", "", "", ""],
    "msa": ["السلام عليكم ورحمة الله", "تحية طيبة", "", "", ""],
    "en": ["Hello", "Hi", "Good morning", "", "", ""],
}

PII_SUFFIX = {
    "gulf": ["رقم ملفي {mrn}", "جوالي {phone}", "هويتي {nid}"],
    "msa": ["رقم الملف {mrn}", "رقم الجوال {phone}", "بريدي {email}"],
    "en": ["my MRN is {mrn}", "my phone is {phone}", "my email is {email}"],
}

_BASE = {
    "appointment": {
        "gulf": [
            ("reschedule", "ابي أأجل موعدي إلى {date} عند {doc}"),
            ("cancel", "لو سمحت الغي موعدي في {dept} ما اقدر اجي"),
            ("book", "ابغى احجز موعد في {dept} {date} ان شاء الله"),
            ("reschedule", "ممكن اغير موعدي مع {doc} لين {date}"),
            ("cancel", "ابي الغي الحجز حقي {date} لاني مسافر"),
            ("book", "وش اقرب موعد متوفر عند {doc} في {dept}"),
        ],
        "msa": [
            ("reschedule", "أرجو تأجيل موعدي إلى {date} مع {doc}"),
            ("cancel", "أود إلغاء موعدي في عيادة {dept} لظرف طارئ"),
            ("book", "أرغب في حجز موعد في عيادة {dept} {date}"),
            ("reschedule", "هل يمكن تعديل موعدي مع {doc} ليصبح {date}"),
            ("cancel", "أرجو إلغاء الحجز المقرر {date} لدى {doc}"),
            ("book", "ما هو أقرب موعد متاح في {dept} لدى {doc}"),
        ],
        "en": [
            ("reschedule", "I need to reschedule my appointment to {date} with {doc}"),
            ("cancel", "Please cancel my appointment in {dept} because I cannot make it"),
            ("book", "I would like to book an appointment in {dept} {date}"),
            ("reschedule", "Can I move my visit with {doc} to {date}"),
            ("cancel", "I want to cancel my booking {date} because I am travelling"),
            ("book", "What is the earliest slot available with {doc} in {dept}"),
        ],
    },
    "billing": {
        "gulf": [
            ("duplicate", "انسحب مني المبلغ مرتين على زيارتي عند {doc}"),
            ("refund", "ابي استرجاع مبلغ {amt} دفعته على زيارة {dept}"),
            ("invoice", "احتاج فاتورة زيارتي {date} للتأمين"),
            ("duplicate", "خصموا علي {amt} مرتين من البطاقة"),
            ("refund", "الغيت موعدي وابي ارجاع المبلغ"),
            ("invoice", "وين اطلع فاتورة الدفع حقت {dept}"),
        ],
        "msa": [
            ("duplicate", "تم خصم المبلغ مرتين من بطاقتي عن زيارة {doc}"),
            ("refund", "أرجو استرداد مبلغ {amt} دفعته مقابل زيارة {dept}"),
            ("invoice", "أحتاج إلى فاتورة رسمية لزيارتي {date} لتقديمها للتأمين"),
            ("duplicate", "لاحظت خصم {amt} مرتين في كشف حسابي"),
            ("refund", "ألغيت موعدي وأرغب في إعادة المبلغ المدفوع"),
            ("invoice", "كيف يمكنني الحصول على فاتورة الدفع الخاصة بعيادة {dept}"),
        ],
        "en": [
            ("duplicate", "I was charged twice for my visit with {doc}"),
            ("refund", "I would like a refund of {amt} that I paid for {dept}"),
            ("invoice", "I need an invoice for my visit {date} for my insurance"),
            ("duplicate", "My card was billed {amt} two times"),
            ("refund", "I cancelled my appointment and want my money back"),
            ("invoice", "Where can I download the payment receipt for {dept}"),
        ],
    },
    "lab_results": {
        "gulf": [
            ("ready", "متى تطلع نتيجة {test} حقتي"),
            ("copy", "ابي نسخة من تقرير {test} للمستشفى ثاني"),
            ("abnormal", "نتيجة {test} طلعت غير طبيعية وش اسوي"),
            ("ready", "سويت {test} {date} وللحين ما وصلتني النتيجة"),
            ("copy", "ممكن ترسلون لي تقرير المختبر على الايميل"),
            ("abnormal", "ليش النتيجة حقت {test} مرتفعة"),
        ],
        "msa": [
            ("ready", "متى ستكون نتيجة {test} جاهزة"),
            ("copy", "أرجو تزويدي بنسخة من تقرير {test}"),
            ("abnormal", "جاءت نتيجة {test} غير طبيعية فما الخطوة التالية"),
            ("ready", "أجريت {test} {date} ولم تصلني النتيجة حتى الآن"),
            ("copy", "هل يمكن إرسال تقرير المختبر إلى بريدي الإلكتروني"),
            ("abnormal", "لماذا ارتفعت نتيجة {test} لدي"),
        ],
        "en": [
            ("ready", "When will my {test} result be ready"),
            ("copy", "Please send me a copy of my {test} report"),
            ("abnormal", "My {test} result came back abnormal what should I do"),
            ("ready", "I did the {test} {date} and still have no result"),
            ("copy", "Can you email me my lab report"),
            ("abnormal", "Why is my {test} result high"),
        ],
    },
    "prescription": {
        "gulf": [
            ("refill", "ابي اجدد وصفة {drug}"),
            ("dosage", "كم مرة اخذ {drug} في اليوم"),
            ("side_effect", "بعد ما اخذت {drug} حسيت بدوخة ووجع بالبطن"),
            ("refill", "خلص علي دواء {drug} ممكن تجديد الوصفة"),
            ("dosage", "هل ينفع اخذ {drug} مع الاكل"),
            ("side_effect", "{drug} يسبب لي حساسية وش البديل"),
        ],
        "msa": [
            ("refill", "أرجو تجديد وصفة {drug} الخاصة بي"),
            ("dosage", "كم مرة يوميا يجب أن أتناول {drug}"),
            ("side_effect", "شعرت بدوار وألم في المعدة بعد تناول {drug}"),
            ("refill", "انتهى دواء {drug} وأحتاج إلى تجديد الوصفة"),
            ("dosage", "هل يمكن تناول {drug} بعد الطعام"),
            ("side_effect", "يسبب لي {drug} حساسية فهل يوجد بديل"),
        ],
        "en": [
            ("refill", "I need to refill my {drug} prescription"),
            ("dosage", "How many times a day should I take {drug}"),
            ("side_effect", "I felt dizzy and had stomach pain after taking {drug}"),
            ("refill", "I ran out of {drug} please renew the prescription"),
            ("dosage", "Can I take {drug} after food"),
            ("side_effect", "{drug} gives me an allergy is there an alternative"),
        ],
    },
}

# 8 extra wordings per (topic, dialect): [train x4, val x2, test x2]. Final order is built below:
#   0-3 base train | 4-7 extra train | 8 base val | 9-10 extra val | 11 base test | 12-13 extra test
_EXTRA = {
    "appointment": {
        "gulf": [
            ("cancel", "الغوا موعدي لو سمحتوا ما اقدر احضر"),
            ("book", "ودي احجز عند دكتور {dept} {date}"),
            ("reschedule", "الموعد حقي مع {doc} ما يناسبني ابي وقت ثاني"),
            ("cancel", "عندي ظرف طاري فالغوا حجزي عند {doc}"),
            ("book", "هل في مواعيد فاضية عند {doc} {date}"),
            ("reschedule", "ممكن تأجلون كشفي لين {date} ظروفي تغيرت"),
            ("reschedule", "ابي ابدل موعدي مع {doc} ليوم ثاني"),
            ("cancel", "الغوا لي كشف {dept} حقي ما بقدر اجي"),
        ],
        "msa": [
            ("cancel", "أرجو إلغاء موعدي فلن أتمكن من الحضور"),
            ("book", "أرغب في حجز موعد لدى دكتور {dept} {date}"),
            ("reschedule", "الموعد المحدد مع {doc} لا يناسبني وأطلب موعدا آخر"),
            ("cancel", "لدي ظرف طارئ لذا أرجو إلغاء حجزي عند {doc}"),
            ("book", "هل توجد مواعيد شاغرة لدى {doc} {date}"),
            ("reschedule", "أطلب تأجيل زيارتي إلى {date} لتعارضها مع عملي"),
            ("reschedule", "أود تغيير موعدي مع {doc} إلى يوم آخر"),
            ("cancel", "أعتذر عن الحضور فأرجو إلغاء كشف {dept} الخاص بي"),
        ],
        "en": [
            ("cancel", "Please cancel my appointment I will not be able to come"),
            ("book", "I want to see a doctor in {dept} {date}"),
            ("reschedule", "My appointment with {doc} does not suit me can I get another time"),
            ("cancel", "I have an emergency so please cancel my booking with {doc}"),
            ("book", "Are there any free slots with {doc} {date}"),
            ("reschedule", "Could you push my visit back to {date} because of work"),
            ("reschedule", "I would like to change my appointment with {doc} to another day"),
            ("cancel", "Sorry I cannot attend so please cancel my {dept} visit"),
        ],
    },
    "billing": {
        "gulf": [
            ("refund", "ابي ارجاع فلوسي لاني ما استفدت من الزيارة"),
            ("invoice", "ممكن ترسلون لي فاتورة {dept} على ايميلي"),
            ("duplicate", "تكرر السحب من حسابي على زيارة {doc}"),
            ("refund", "دفعت {amt} غلط وابي ارجعهم"),
            ("duplicate", "المبلغ انسحب مرتين من البطاقة يوم زرت {dept}"),
            ("invoice", "ابي فاتورة بمبلغ {amt} عشان اقدمها للتأمين"),
            ("duplicate", "انسحب {amt} من حسابي مرتين ليش"),
            ("refund", "ابي استرجع {amt} من زيارة {doc} الملغية"),
        ],
        "msa": [
            ("refund", "أطلب إعادة أموالي لأنني لم أستفد من الزيارة"),
            ("invoice", "أرجو إرسال فاتورة عيادة {dept} إلى بريدي"),
            ("duplicate", "تكرر السحب من حسابي مقابل زيارة {doc}"),
            ("refund", "دفعت مبلغ {amt} عن طريق الخطأ وأطلب إعادته"),
            ("duplicate", "سحب المبلغ من بطاقتي مرتين عند زيارة {dept}"),
            ("invoice", "أحتاج فاتورة بمبلغ {amt} لتقديمها لشركة التأمين"),
            ("duplicate", "تم سحب {amt} من حسابي مرتين فما السبب"),
            ("refund", "أرغب في استرداد {amt} عن زيارة {doc} الملغاة"),
        ],
        "en": [
            ("refund", "I want my money back because I did not use the visit"),
            ("invoice", "Please email me the invoice for my {dept} visit"),
            ("duplicate", "The payment for my visit with {doc} was taken twice"),
            ("refund", "I paid {amt} by mistake and need it returned"),
            ("duplicate", "My card was debited two times when I visited {dept}"),
            ("invoice", "I need an invoice of {amt} to give to my insurance company"),
            ("duplicate", "I see {amt} taken from my account twice why"),
            ("refund", "I would like {amt} back for the cancelled visit with {doc}"),
        ],
    },
    "lab_results": {
        "gulf": [
            ("copy", "ابي ورقة النتايج حقتي مطبوعة"),
            ("abnormal", "الدكتور قال {test} عندي مو طبيعي وش المطلوب"),
            ("ready", "كم باقي على نتيجة {test} اللي سويتها"),
            ("copy", "ارسلوا لي نسخة {test} واتساب لو سمحتوا"),
            ("ready", "وين وصلت نتيجة {test} اللي اخذتوا عينتها {date}"),
            ("abnormal", "قيمة {test} عندي مرتفعة وخايف منها"),
            ("ready", "ابي اعرف متى تجهز {test} اللي سويتها {date}"),
            ("copy", "ممكن تعطوني تقرير {test} بالانجليزي"),
        ],
        "msa": [
            ("copy", "أرغب في الحصول على نسخة مطبوعة من نتائجي"),
            ("abnormal", "أخبرني الطبيب أن {test} غير طبيعي فماذا أفعل"),
            ("ready", "كم تبقى من الوقت لصدور نتيجة {test}"),
            ("copy", "أرجو إرسال نسخة من {test} عبر الواتساب"),
            ("ready", "أين وصلت نتيجة {test} التي أخذت عينتها {date}"),
            ("abnormal", "قيمة {test} لدي مرتفعة وأشعر بالقلق"),
            ("ready", "أود معرفة موعد جاهزية نتيجة {test} التي أجريتها {date}"),
            ("copy", "أحتاج تقريرا بالإنجليزية عن {test}"),
        ],
        "en": [
            ("copy", "I would like a printed copy of my results"),
            ("abnormal", "My doctor said my {test} is not normal what now"),
            ("ready", "How much longer until my {test} result is out"),
            ("copy", "Please send me my {test} report on WhatsApp"),
            ("ready", "Where is the {test} result for the sample taken {date}"),
            ("abnormal", "My {test} value is high and I am worried"),
            ("ready", "I want to know when my {test} from {date} will be ready"),
            ("copy", "I need an English report of my {test}"),
        ],
    },
    "prescription": {
        "gulf": [
            ("dosage", "متى احسن وقت اخذ حبوب الضغط"),
            ("side_effect", "الدوا حقي سبب لي طفح جلدي وش اسوي"),
            ("refill", "وصفة {drug} خلصت وابي واحدة جديدة"),
            ("dosage", "كم حبة من {drug} اخذ بالمرة"),
            ("refill", "ممكن تعيدون لي صرف {drug} لهالشهر"),
            ("side_effect", "من بدأت {drug} وانا اصحى متعب وراسي يوجعني"),
            ("refill", "اباكم تجددون {drug} قبل لا يخلص"),
            ("dosage", "هل اقدر اخذ {drug} مع القهوة"),
        ],
        "msa": [
            ("dosage", "ما هو أفضل وقت لتناول دواء الضغط"),
            ("side_effect", "تسبب لي الدواء بطفح جلدي فماذا أفعل"),
            ("refill", "انتهت وصفة {drug} وأحتاج وصفة جديدة"),
            ("dosage", "كم حبة من {drug} أتناول في المرة الواحدة"),
            ("refill", "أرجو إعادة صرف {drug} لهذا الشهر"),
            ("side_effect", "منذ بدأت {drug} وأنا أشعر بالتعب والصداع"),
            ("refill", "أريد تجديد {drug} قبل أن ينتهي"),
            ("dosage", "هل يمكنني تناول {drug} مع القهوة"),
        ],
        "en": [
            ("dosage", "What is the best time to take my blood pressure pills"),
            ("side_effect", "This medication gave me a skin rash what should I do"),
            ("refill", "My {drug} prescription is finished and I need a new one"),
            ("dosage", "How many tablets of {drug} do I take at once"),
            ("refill", "Please reissue my {drug} for this month"),
            ("side_effect", "Since starting {drug} I feel tired and have headaches"),
            ("refill", "I want to renew {drug} before it runs out"),
            ("dosage", "Can I take {drug} with coffee"),
        ],
    },
}

TEMPLATES = {
    t: {st: _BASE[t][st][:4] + _EXTRA[t][st][:4] + [_BASE[t][st][4]] + _EXTRA[t][st][4:6]
        + [_BASE[t][st][5]] + _EXTRA[t][st][6:8] for st in _BASE[t]}
    for t in _BASE
}


# What the case-management team did (shown as the "similar resolved case").
RESOLUTIONS = {
    ("appointment", "reschedule"): ("تم نقل الموعد وإرسال تأكيد برسالة نصية.", "Appointment moved and a confirmation SMS sent."),
    ("appointment", "cancel"): ("تم إلغاء الحجز وتحرير الموعد.", "Booking cancelled and the slot released."),
    ("appointment", "book"): ("تم حجز أقرب موعد متاح للمريض.", "Earliest available slot booked for the patient."),
    ("billing", "duplicate"): ("تم التحقق من الخصم المزدوج وبدء استرجاع المبلغ المكرر.", "Duplicate charge verified, refund of the extra amount started."),
    ("billing", "refund"): ("تمت الموافقة على الاسترجاع خلال 7 أيام عمل.", "Refund approved within 7 working days."),
    ("billing", "invoice"): ("تم إرسال الفاتورة الرسمية إلى البريد المسجل.", "Official invoice sent to the registered email."),
    ("lab_results", "ready"): ("النتيجة جاهزة خلال 48 ساعة وتصل برسالة نصية.", "Result ready within 48 hours and sent by SMS."),
    ("lab_results", "copy"): ("تم إرسال نسخة من التقرير عبر البوابة الإلكترونية.", "Report copy sent through the patient portal."),
    ("lab_results", "abnormal"): ("تم تحويل الحالة للطبيب المعالج لمراجعة النتيجة.", "Case forwarded to the treating physician for review."),
    ("prescription", "refill"): ("تم تجديد الوصفة وإرسالها إلى الصيدلية.", "Prescription renewed and sent to the pharmacy."),
    ("prescription", "dosage"): ("تم توضيح الجرعة من الصيدلي المسؤول.", "Dosage clarified by the pharmacist on duty."),
    ("prescription", "side_effect"): ("تم تحويل المريض للطبيب لتقييم الأعراض.", "Patient referred to the doctor to assess symptoms."),
}
