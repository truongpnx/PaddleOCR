// trace_skeleton.cpp
// Trace skeletonization result into polylines
//
// Lingdong Huang 2020

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <string>
#include <climits>

//================================
// ENUMS
//================================
#define HORIZONTAL 1
#define VERTICAL 2

//================================
// PARAMS
//================================
#define CHUNK_SIZE 10           // the chunk size
#define SAVE_RECTS 1            // additionally save bounding rects of chunks (for visualization)
#define MAX_ITER 999            // maximum number of iterations


struct skeleton_tracer_t {
  //================================
  // GLOBALS
  //================================
  typedef unsigned char uchar;
  uchar* im; // the image
  int W;     // width
  int H;     // height

  skeleton_tracer_t(){
    im = NULL;
    rects.head = NULL;
    rects.tail = NULL;
  }

  //================================
  // DATASTRUCTURES
  //================================

  typedef struct _point_t {
    int x;
    int y;
    struct _point_t * next;
  } point_t;

  typedef struct _polyline_t {
    point_t* head;
    point_t* tail;
    struct _polyline_t* prev;
    struct _polyline_t* next;
    int size;
  } polyline_t;


  typedef struct _rect_t {
    int x;
    int y;
    int w;
    int h;
    struct _rect_t* next;
  } rect_t;

  struct _rects_t{
    rect_t* head;
    rect_t* tail;
  } rects;

  //================================
  // DATASTRUCTURE IMPLEMENTATION
  //================================

  polyline_t* new_polyline(){
    polyline_t* q0 = (polyline_t*)malloc(sizeof(polyline_t));
    q0->head = NULL;
    q0->tail = NULL;
    q0->prev = NULL;
    q0->next = NULL;
    q0->size = 0;
    return q0;
  }
  std::string print_polyline(polyline_t* q){
    std::string str = "";
    if (!q){
      return str;
    }
    point_t* jt = q->head;
    while(jt){
      str += std::to_string(jt->x)+","+std::to_string(jt->y)+" ";
      jt = jt->next;
    }
    return str;
  }
  std::string print_polylines(polyline_t* q){
    std::string str = "";
    if (!q){
      return str;
    }
    polyline_t* it = q;
    while(it){
      point_t* jt = it->head;
      while(jt){
        str += std::to_string(jt->x)+","+std::to_string(jt->y)+" ";
        jt = jt->next;
      }
      str += "\n";
      it = it->next;
    }
    return str;
  }
  void destroy_polylines(polyline_t* q){
    if (!q){
      return;
    }
    polyline_t* it = q;
    while(it){
      polyline_t* lt = it->next;
      point_t* jt = it->head;
      while(jt){
        point_t* kt = jt->next;
        free(jt);
        jt = kt;
      }
      free(it);
      it = lt;
    }
  }

  void reverse_polyline(polyline_t* q){
    if (!q || (q->size < 2)){
      return;
    }
    q->tail->next = q->head;
    point_t* it0 = q->head;
    point_t* it1 = it0->next;
    point_t* it2 = it1->next;
    int i; for (i = 0; i < q->size-1; i++){
        it1->next = it0;
        it0 = it1;
        it1 = it2;
        it2 = it2->next;
    }
    point_t* q_head = q->head;
    q->head = q->tail;
    q->tail = q_head;
    q->tail->next = NULL;
  }

  void cat_tail_polyline(polyline_t* q0, polyline_t* q1){
    if (!q1){
      return;
    }
    if (!q0){
      *q0 = *new_polyline();
    }
    if (!q0->head){
      q0->head = q1->head;
      q0->tail = q1->tail;
      return;
    }
    q0->tail->next = q1->head;
    q0->tail = q1->tail;
    q0->size += q1->size;
    q0->tail->next = NULL;
  }

  void cat_head_polyline(polyline_t* q0, polyline_t* q1){
    if (!q1){
      return;
    }
    if (!q0){
      *q0 = *new_polyline();
    }
    if (!q1->head){
      return;
    }
    if (!q0->head){
      q0->head = q1->head;
      q0->tail = q1->tail;
      return;
    }
    q1->tail->next = q0->head;
    q0->head = q1->head;
    q0->size += q1->size;
    q0->tail->next = NULL;
  }

  void add_point_to_polyline(polyline_t* q, int x, int y){
    point_t* p = (point_t*)malloc(sizeof(point_t));
    p->x = x;
    p->y = y;
    p->next = NULL;
    if (!q->head){
      q->head = p;
      q->tail = p;
    }else{
      q->tail->next = p;
      q->tail = p;
    }
    q->size++;
  }

  polyline_t* prepend_polyline(polyline_t* q0, polyline_t* q1){
    if (!q0){
      return q1;
    }
    q1->next = q0;
    q0->prev = q1;
    return q1;
  }

  std::string print_rects(){
    std::string str;
    rect_t* it = rects.head;
    while(it){
      str += std::to_string(it->x)+","+std::to_string(it->y)+","+std::to_string(it->w)+","+std::to_string(it->h)+"\n";
      it = it->next;
    }
    return str;
  }

  void destroy_rects(){
    rect_t* it = rects.head;
    while(it){
      rect_t* jt = it->next;
      free(it);
      it = jt;
    }
    rects.head = NULL;
    rects.tail = NULL;
  }

  void add_rect(int x, int y, int w, int h){
    #if SAVE_RECTS
      rect_t* r = (rect_t*)malloc(sizeof(rect_t));
      r->x = x;
      r->y = y;
      r->w = w;
      r->h = h;
      r->next = NULL;
      if (!rects.head){
        rects.head = r;
        rects.tail = r;
      }else{
        rects.tail->next = r;
        rects.tail = r;
      }
    #endif
  }

  //================================
  // RASTER SKELETONIZATION
  //================================
  // Binary image thinning (skeletonization) in-place.
  // Implements Zhang-Suen algorithm.
  // http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf
  bool thinning_zs_iteration(int iter) {
    bool diff = false;
    for (int i = 1; i < H-1; i++){
      for (int j = 1; j < W-1; j++){
        int p2 = im[(i-1)*W+j]   & 1;
        int p3 = im[(i-1)*W+j+1] & 1;
        int p4 = im[(i)*W+j+1]   & 1;
        int p5 = im[(i+1)*W+j+1] & 1;
        int p6 = im[(i+1)*W+j]   & 1;
        int p7 = im[(i+1)*W+j-1] & 1;
        int p8 = im[(i)*W+j-1]   & 1;
        int p9 = im[(i-1)*W+j-1] & 1;
        
        int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
          (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
          (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
          (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
        int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
        int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
        int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
        if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
          im[i*W+j] |= 2;
      }
    }
    for (int i = 0; i < H*W; i++){
      int marker = im[i]>>1;
      int old = im[i]&1;
      im[i] = old & (!marker);
      if ((!diff) && (im[i] != old)){
        diff = true;
      }
    }
    return diff;
  };
  void thinning_zs(){
    bool diff = true;
    do {
      diff &= thinning_zs_iteration(0);
      diff &= thinning_zs_iteration(1);
    }while (diff);
  }

  //================================
  // MAIN ALGORITHM
  //================================

  // check if a region has any white pixel
  int not_empty(int x, int y, int w, int h){
    for (int i = y; i < y+h; i++){
      for (int j = x; j < x+w; j++){
        if (im[i*W+j]){
          return 1;
        }
      }
    }
    return 0;
  }

  /**merge ith fragment of second chunk to first chunk
   * @param c0   fragments from  first  chunk
   * @param c1i  ith fragment of second chunk
   * @param sx   (x or y) coordinate of the seam
   * @param isv  is vertical, not horizontal?
   * @param mode 2-bit flag, 
   *             MSB = is matching the left (not right) end of the fragment from first  chunk
   *             LSB = is matching the right (not left) end of the fragment from second chunk
   * @return     matching successful?             
   */
  int merge_impl(polyline_t* c0, polyline_t* c1i, int sx, int isv, int mode){
    int b0 = (mode >> 1 & 1)>0; // match c0 left
    int b1 = (mode >> 0 & 1)>0; // match c1 left
    polyline_t* c0j = NULL;
    int md = 4; // maximum offset to be regarded as continuous

    point_t* p1 = b1 ? c1i->head : c1i->tail;

    if (abs((isv?(p1->y):(p1->x))-sx)>0){ // not on the seam, skip
      return 0;
    }
    // find the best match
    polyline_t* it = c0;
    while (it){
      point_t* p0 = b0?(it->head):(it->tail);
      if (abs((isv?(p0->y):(p0->x))-sx)>1){ // not on the seam, skip
        it = it->next;
        continue;
      }
      int d = abs((isv?(p0->x):(p0->y)) - (isv?(p1->x):(p1->y)));
      if (d < md){
        c0j = it;
        md = d;
      }
      it = it->next;
    }

    if (c0j){ // best match is good enough, merge them
      if (b0 && b1){
        reverse_polyline(c1i);
        cat_head_polyline(c0j,c1i);
      }else if (!b0 && b1){
        cat_tail_polyline(c0j,c1i);
      }else if (b0 && !b1){
        cat_head_polyline(c0j,c1i);
      }else {
        reverse_polyline(c1i);
        cat_tail_polyline(c0j,c1i);
      }
      return 1;    
    }
    return 0;
  }

  /**merge fragments from two chunks
   * @param c0   fragments from first  chunk
   * @param c1   fragments from second chunk
   * @param sx   (x or y) coordinate of the seam
   * @param dr   merge direction, HORIZONTAL or VERTICAL?
   */
  polyline_t* merge_frags(polyline_t* c0, polyline_t* c1, int sx, int dr){
    if (!c0){
      return c1;
    }
    if (!c1){
      return c0;
    }
    polyline_t* it = c1;
    while(it){
      polyline_t* tmp = it->next;
      if (dr == HORIZONTAL){
        if (merge_impl(c0,it,sx,0,1))goto rem;
        if (merge_impl(c0,it,sx,0,3))goto rem;
        if (merge_impl(c0,it,sx,0,0))goto rem;
        if (merge_impl(c0,it,sx,0,2))goto rem;
      }else{
        if (merge_impl(c0,it,sx,1,1))goto rem;
        if (merge_impl(c0,it,sx,1,3))goto rem;
        if (merge_impl(c0,it,sx,1,0))goto rem;
        if (merge_impl(c0,it,sx,1,2))goto rem;      
      }
      goto next;
      rem:
      if (!it->prev){
        c1 = it->next;
        if (it->next){
          it->next->prev = NULL;
        }
      }else{
        it->prev->next = it->next;
        if (it->next){
          it->next->prev = it->prev;
        }
      }
      free(it);
      next:
      it = tmp;
    }
    it = c1;
    while(it){
      polyline_t* tmp = it->next;
      it->prev = NULL;
      it->next = NULL;
      c0 = prepend_polyline(c0,it);
      it = tmp;
    }
    return c0;
  }

  /**recursive bottom: turn chunk into polyline fragments;
   * look around on 4 edges of the chunk, and identify the "outgoing" pixels;
   * add segments connecting these pixels to center of chunk;
   * apply heuristics to adjust center of chunk
   *
   * @param x    left of   chunk
   * @param y    top of    chunk
   * @param w    width of  chunk
   * @param h    height of chunk
   * @return     the polyline fragments
   */
  polyline_t* chunk_to_frags(int x, int y, int w, int h){
    polyline_t* frags = NULL;
    int fsize = 0;
    int on = 0; // to deal with strokes thicker than 1px
    int li=-1, lj=-1;
    
    // walk around the edge clockwise
    for (int k = 0; k < h+h+w+w-4; k++){
      int i, j;
      if (k < w){
        i = y+0; j = x+k;
      }else if (k < w+h-1){
        i = y+k-w+1; j = x+w-1;
      }else if (k < w+h+w-2){
        i = y+h-1; j = x+w-(k-w-h+3); 
      }else{
        i = y+h-(k-w-h-w+4); j = x+0;
      }
      if (im[i*W+j]){ // found an outgoing pixel
        if (!on){     // left side of stroke
          on = 1;
          polyline_t* f = new_polyline();
          add_point_to_polyline(f, j, i);
          add_point_to_polyline(f, x+w/2,y+h/2);
          frags = prepend_polyline(frags,f);
          fsize ++;
        }
      }else{
        if (on){// right side of stroke, average to get center of stroke
          frags->head->x = (frags->head->x+lj)/2;
          frags->head->y = (frags->head->y+li)/2;
          on = 0;
        }
      }
      li = i;
      lj = j;
    }
    if (fsize == 2){ // probably just a line, connect them
      polyline_t* f = new_polyline();
      add_point_to_polyline(f,frags->head->x,frags->head->y);
      add_point_to_polyline(f,frags->next->head->x,frags->next->head->y);
      destroy_polylines(frags);
      frags = f;
    }else if (fsize > 2){ // it's a crossroad, guess the intersection
      int ms = 0;
      int mi = -1;
      int mj = -1;
      // use convolution to find brightest blob
      for (int i = y+1; i < y+h-1; i++){
        for (int j = x+1; j < x+w-1; j++){
          int s = 
            (im[i*W-W+j-1]) + (im[i*W-W+j]) + (im[i*W-W+j-1+1])+
            (im[i*W+j-1]  ) +   (im[i*W+j]) +   (im[i*W+j+1]  )+
            (im[i*W+W+j-1]) + (im[i*W+W+j]) + (im[i*W+W+j+1]  );
          if (s > ms){
            mi = i;
            mj = j;
            ms = s;
          }else if (s == ms && abs(j-(x+w/2))+abs(i-(y+h/2)) < abs(mj-(x+w/2))+abs(mi-(y+h/2))){
            mi = i;
            mj = j;
            ms = s;
          }
        }
      }
      if (mi != -1){
        polyline_t* it = frags;
        while(it){
          it->tail->x = mj;
          it->tail->y = mi;
          it = it->next;
        }
      }
    }
    return frags;
  }


  /**Trace skeleton from thinning result.
   * Algorithm:
   * 1. if chunk size is small enough, reach recursive bottom and turn it into segments
   * 2. attempt to split the chunk into 2 smaller chunks, either horizontall or vertically;
   *    find the best "seam" to carve along, and avoid possible degenerate cases
   * 3. recurse on each chunk, and merge their segments
   *
   * @param x       left of   chunk
   * @param y       top of    chunk
   * @param w       width of  chunk
   * @param h       height of chunk
   * @param iter    current iteration
   * @return        an array of polylines
  */
  polyline_t* trace_skeleton(int x, int y, int w, int h, int iter){
    // printf("_%d %d %d %d %d\n",x,y,w,h,iter);

    polyline_t* frags = NULL;
    
    if (iter >= MAX_ITER){ // gameover
      return frags;
    }
    if (w <= CHUNK_SIZE && h <= CHUNK_SIZE){ // recursive bottom
      frags = chunk_to_frags(x,y,w,h);
      return frags;
    }
   
    int ms = INT_MAX; // number of white pixels on the seam, less the better
    int mi = -1; // horizontal seam candidate
    int mj = -1; // vertical   seam candidate
    
    if (h > CHUNK_SIZE){ // try splitting top and bottom
      for (int i = y+3; i < y+h-3; i++){
        if (im[i*W+x] ||im[(i-1)*W+x] ||im[i*W+x+w-1] ||im[(i-1)*W+x+w-1]){
          continue;
        }
        int s = 0;
        for (int j = x; j < x+w; j++){
          s += im[i*W+j];
          s += im[(i-1)*W+j];
        }
        if (s < ms){
          ms = s; mi = i;
        }else if (s == ms && abs(i-(y+h/2))<abs(mi-(y+h/2))){
          // if there is a draw (very common), we want the seam to be near the middle
          // to balance the divide and conquer tree
          ms = s; mi = i;
        }
      }
    }

    if (w > CHUNK_SIZE){ // same as above, try splitting left and right
      for (int j = x+3; j < x+w-3; j++){
        if (im[W*y+j]||im[W*(y+h)-W+j]||im[W*y+j-1]||im[W*(y+h)-W+j-1]){
          continue;
        }
        int s = 0;
        for (int i = y; i < y+h; i++){
          s += im[i*W+j]?1:0;
          s += im[i*W+j-1]?1:0;
        }
        if (s < ms){
          ms = s;
          mi = -1; // horizontal seam is defeated
          mj = j;
        }else if (s == ms && abs(j-(x+w/2))<abs(mj-(x+w/2))){
          ms = s;
          mi = -1;
          mj = j;
        }
      }
    }

    int L0=-1; int L1; int L2; int L3;
    int R0=-1; int R1; int R2; int R3;
    int dr = 0;
    int sx;
    if (h > CHUNK_SIZE && mi != -1){ // split top and bottom
      L0 = x; L1 = y;  L2 = w; L3 = mi-y;
      R0 = x; R1 = mi; R2 = w; R3 = y+h-mi;
      dr = VERTICAL;
      sx = mi;
    }else if (w > CHUNK_SIZE && mj != -1){ // split left and right
      L0 = x; L1 = y; L2 = mj-x; L3 = h;
      R0 = mj;R1 = y; R2 =x+w-mj;R3 = h;
      dr = HORIZONTAL;
      sx = mj;
    }

    if (dr!=0 && not_empty(L0,L1,L2,L3)){ // if there are no white pixels, don't waste time
      #if SAVE_RECTS
        add_rect(L0,L1,L2,L3);
      #endif
      frags = trace_skeleton(L0,L1,L2,L3,iter+1);
    }
    if (dr!=0 && not_empty(R0,R1,R2,R3)){
      #if SAVE_RECTS
        add_rect(R0,R1,R2,R3);
      #endif
      frags = merge_frags(frags, trace_skeleton(R0,R1,R2,R3,iter+1),sx,dr);
    }

    if (mi == -1 && mj == -1){ // splitting failed! do the recursive bottom instead
      frags = chunk_to_frags(x,y,w,h);
    }

    return frags;
  }


  //================================
  // GUI/IO
  //================================
  void print_bitmap(){
    for (int i = 0; i < H; i++){
      for (int j = 0; j < W; j++){
        printf("%d",im[i*W+j]);
      }
      printf("\n");
    }
  }

  char* trace(char* img, int w, int h){

    W = w;
    H = h;
    if (im){
      free(im);
    }
    destroy_rects();

    im = (uchar*)img;

    // print_bitmap();
    thinning_zs();
    // print_bitmap();
    
    polyline_t* p = (polyline_t*)trace_skeleton(0,0,W,H,0);
    std::string str = "POLYLINES:\n"+print_polylines(p)+"RECTS:\n"+print_rects();
    destroy_polylines(p);

    // printf("%s\n",str.c_str());

    char * writable = (char*)malloc((str.size()+1)*sizeof(char));
    std::copy(str.begin(), str.end(), writable);
    writable[str.size()] = '\0';
    return writable;
  }

  void destroy(){
    if (im){
      free(im);
    }
    destroy_rects();
  }

};


